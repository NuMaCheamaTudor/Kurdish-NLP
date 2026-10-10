"""Post-hoc temperature scaling and abstention-threshold selection.

Both baselines are softmax classifiers, so ``log p`` equals the logits up to a
per-row constant and temperature scaling ``softmax(log p / T)`` applies to both
without access to model internals. Temperature scaling never changes the arg-max
(accuracy and F1 are unchanged); it only reshapes confidence.

Protocol: temperatures and thresholds are fitted only on ``dev_calib``, which
took no part in fitting model parameters (train) or selecting hyperparameters
(dev_tune). Whether calibration is adopted at all is decided by group-disjoint
cross-fitting inside ``dev_calib``, so the raw-versus-calibrated comparison is
out-of-sample. Calibrated scores remain estimates, not guaranteed probabilities
of correctness, and they are fitted to the dev distribution only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from kurdish_nlp.langid.experiments.config import AbstentionSpec, CalibrationSpec
from kurdish_nlp.langid.experiments.evaluation import (
    brier_score,
    expected_calibration_error,
    log_loss,
    rnd,
    top_two,
    vectorized_acceptance,
)


def apply_temperature(
    proba: np.ndarray, temperature: float, floor: float
) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    logits = np.log(np.maximum(np.asarray(proba, dtype=np.float64), floor))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    proba: np.ndarray, y_true: Sequence[int], spec: CalibrationSpec
) -> float:
    """Minimise negative log-likelihood over log(T) within configured bounds."""

    true = np.asarray(y_true, dtype=np.int64)
    low, high = spec.temperature_bounds

    def objective(log_t: float) -> float:
        return log_loss(
            apply_temperature(proba, math.exp(log_t), spec.probability_floor),
            true,
            spec.probability_floor,
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(math.exp(result.x))


def _scores(proba: np.ndarray, true: np.ndarray, spec: CalibrationSpec) -> dict:
    ece, _ = expected_calibration_error(proba, true, spec.ece_bins)
    return {
        "log_loss": rnd(log_loss(proba, true, spec.probability_floor)),
        "brier": rnd(brier_score(proba, true)),
        "ece": rnd(ece),
    }


def calibrate(
    proba: np.ndarray,
    y_true: Sequence[int],
    folds: Sequence[str],
    spec: CalibrationSpec,
) -> dict[str, Any]:
    """Cross-fit the adoption decision, then fit the final temperature.

    For every fold, T is fitted on the other folds and applied to it; the
    concatenated out-of-fold probabilities are compared with the raw ones.
    Temperature scaling is adopted only if out-of-fold log loss improves. The
    final T is then fitted on all of dev_calib.
    """

    true = np.asarray(y_true, dtype=np.int64)
    fold_array = np.asarray(folds)
    out_of_fold = np.empty_like(np.asarray(proba, dtype=np.float64))
    fold_temperatures = {}
    for fold in sorted(set(folds)):
        held = fold_array == fold
        temperature = fit_temperature(proba[~held], true[~held], spec)
        fold_temperatures[fold] = rnd(temperature)
        out_of_fold[held] = apply_temperature(
            proba[held], temperature, spec.probability_floor
        )
    raw = _scores(proba, true, spec)
    crossfit = _scores(out_of_fold, true, spec)
    adopt = crossfit["log_loss"] < raw["log_loss"]
    final_temperature = fit_temperature(proba, true, spec) if adopt else 1.0
    return {
        "method": spec.method,
        "fit_role": spec.fit_role,
        "records": len(true),
        "raw": raw,
        "crossfit_temperature_scaling": crossfit,
        "fold_temperatures": fold_temperatures,
        "adopted": bool(adopt),
        "temperature": rnd(final_temperature),
        "decision": (
            "temperature scaling adopted: out-of-fold log loss improved"
            if adopt
            else "raw scores kept: out-of-fold log loss did not improve"
        ),
    }


def threshold_grid(
    proba: np.ndarray,
    y_true: Sequence[int],
    letters: np.ndarray,
    spec: AbstentionSpec,
) -> list[dict[str, Any]]:
    true = np.asarray(y_true, dtype=np.int64)
    first, _, _, _ = top_two(proba)
    correct = first == true
    rows = []
    for confidence in spec.confidence_grid:
        for margin in spec.margin_grid:
            accepted = vectorized_acceptance(
                proba,
                letters,
                confidence_threshold=confidence,
                margin_threshold=margin,
                minimum_letter_count=spec.minimum_letter_count,
            )
            covered = int(accepted.sum())
            rows.append(
                {
                    "confidence_threshold": confidence,
                    "margin_threshold": margin,
                    "coverage": rnd(covered / len(true)),
                    "selective_accuracy": rnd(
                        (correct & accepted).sum() / covered if covered else None
                    ),
                }
            )
    return rows


def select_thresholds(
    proba: np.ndarray,
    y_true: Sequence[int],
    letters: np.ndarray,
    spec: AbstentionSpec,
) -> dict[str, Any]:
    """Maximise coverage subject to the selective-accuracy target on dev_calib."""

    grid = threshold_grid(proba, y_true, letters, spec)
    feasible = [
        row
        for row in grid
        if row["selective_accuracy"] is not None
        and row["selective_accuracy"] >= spec.target_selective_accuracy
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda r: (
                r["coverage"],
                -r["confidence_threshold"],
                -r["margin_threshold"],
            ),
        )
        rule = "max coverage with selective accuracy >= target"
    else:
        fallback = [
            row
            for row in grid
            if row["selective_accuracy"] is not None
            and row["coverage"] >= spec.minimum_coverage
        ]
        if not fallback:
            raise RuntimeError("no abstention threshold satisfies the fallback rule")
        chosen = max(
            fallback,
            key=lambda r: (
                r["selective_accuracy"],
                r["coverage"],
                -r["confidence_threshold"],
                -r["margin_threshold"],
            ),
        )
        rule = "fallback: max selective accuracy with coverage >= minimum"
    return {
        "fit_role": spec.fit_role,
        "target_selective_accuracy": spec.target_selective_accuracy,
        "minimum_coverage": spec.minimum_coverage,
        "minimum_letter_count": spec.minimum_letter_count,
        "rule": rule,
        "target_met": bool(feasible),
        "confidence_threshold": chosen["confidence_threshold"],
        "margin_threshold": chosen["margin_threshold"],
        "dev_calib_coverage": chosen["coverage"],
        "dev_calib_selective_accuracy": chosen["selective_accuracy"],
        "grid": grid,
    }

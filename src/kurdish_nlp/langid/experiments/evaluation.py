"""Shared, deterministic evaluator for every model family.

Every function takes integer class indices in the experiment's configured label
order and probability matrices whose columns follow that same order. Nothing
here depends on a model library, so both baselines are scored by identical code.

Conventions:

* precision, recall and F1 use ``zero_division = 0`` (as scikit-learn's default);
* macro averages run over ``labels`` that have support, unless ``all_labels``
  is requested (used for the seven-class internal evaluation);
* model scores are estimates; calibrated or not, they are not guaranteed
  probabilities of correctness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from kurdish_nlp.langid.normalization import lexical_letter_count
from kurdish_nlp.langid.reliability import ReliabilityPolicy
from kurdish_nlp.langid.schemas import LanguageCode, Prediction

DECIMALS = 6


def rnd(value: float | None) -> float | None:
    """Round for stable, diff-friendly JSON; ``None`` stays ``None``."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("metrics must be finite")
    return round(value, DECIMALS)


def check_probabilities(proba: np.ndarray, n_labels: int) -> np.ndarray:
    matrix = np.asarray(proba, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n_labels:
        raise ValueError(f"probabilities must have shape (n, {n_labels})")
    if not np.all(np.isfinite(matrix)) or matrix.min(initial=0.0) < 0:
        raise ValueError("probabilities must be finite and non-negative")
    sums = matrix.sum(axis=1)
    if matrix.shape[0] and not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to 1")
    return matrix


def renormalize(proba: np.ndarray) -> np.ndarray:
    matrix = np.clip(np.asarray(proba, dtype=np.float64), 0.0, None)
    sums = matrix.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("a probability row has no mass")
    return matrix / sums


def ranked(proba: np.ndarray) -> np.ndarray:
    """Column indices by descending probability; ties keep label order."""

    return np.argsort(-proba, axis=1, kind="stable")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """Rows are true labels, columns predictions, both in label order."""

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred differ in length")
    if len(y_true) and (y_true.min() < 0 or y_true.max() >= n):
        raise ValueError("y_true outside the label range")
    if len(y_pred) and (y_pred.min() < 0 or y_pred.max() >= n):
        raise ValueError("y_pred outside the prediction range")
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def _prf_from_confusion(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0
    )
    return precision, recall, f1, support


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    _, _, f1, support = _prf_from_confusion(confusion_matrix(y_true, y_pred, n))
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    *,
    all_labels: bool = True,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro/weighted P/R/F1 and per-class values."""

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    n = len(labels)
    matrix = confusion_matrix(true, pred, n)
    precision, recall, f1, support = _prf_from_confusion(matrix)
    active = np.ones(n, dtype=bool) if all_labels else support > 0
    total = support.sum()
    weights = support / total if total else support
    return {
        "records": int(total),
        "accuracy": rnd(np.trace(matrix) / total if total else 0.0),
        "balanced_accuracy": rnd(recall[support > 0].mean() if total else 0.0),
        "macro_precision": rnd(precision[active].mean()),
        "macro_recall": rnd(recall[active].mean()),
        "macro_f1": rnd(f1[active].mean()),
        "weighted_f1": rnd((f1 * weights).sum()),
        "macro_over": [label for label, keep in zip(labels, active) if keep],
        "per_class": {
            label: {
                "precision": rnd(precision[i]),
                "recall": rnd(recall[i]),
                "f1": rnd(f1[i]),
                "support": int(support[i]),
                "predicted": int(matrix[:, i].sum()),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": list(labels),
            "rows": "true",
            "columns": "predicted",
            "matrix": matrix.tolist(),
        },
    }


def log_loss(proba: np.ndarray, y_true: np.ndarray, floor: float) -> float:
    picked = proba[np.arange(len(y_true)), y_true]
    return float(-np.log(np.maximum(picked, floor)).mean())


def brier_score(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score: mean over records of sum_k (p_k - 1[y=k])^2."""

    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def expected_calibration_error(
    proba: np.ndarray, y_true: np.ndarray, bins: int
) -> tuple[float, list[dict[str, Any]]]:
    """Top-label ECE with equal-width confidence bins on (0, 1].

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|. Bin b holds
    confidences in ((b-1)/B, b/B]; a confidence of exactly 0 goes to bin 1.
    """

    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(np.float64)
    index = np.clip(np.ceil(confidence * bins).astype(np.int64) - 1, 0, bins - 1)
    total = len(y_true)
    ece = 0.0
    table = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            ece += count / total * abs(acc - conf)
        else:
            acc = conf = None
        table.append(
            {
                "bin": b + 1,
                "lower": rnd(b / bins),
                "upper": rnd((b + 1) / bins),
                "count": count,
                "accuracy": rnd(acc),
                "mean_confidence": rnd(conf),
            }
        )
    return ece, table


def probability_report(
    proba: np.ndarray,
    y_true: Sequence[int],
    *,
    floor: float,
    ece_bins: int,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    order = ranked(proba)
    ece, table = expected_calibration_error(proba, true, ece_bins)
    top2 = (order[:, :2] == true[:, None]).any(axis=1)
    return {
        "log_loss": rnd(log_loss(proba, true, floor)),
        "brier": rnd(brier_score(proba, true)),
        "ece": rnd(ece),
        "ece_bins": ece_bins,
        "top2_accuracy": rnd(top2.mean() if len(true) else 0.0),
        "mean_top1_confidence": rnd(proba.max(axis=1).mean() if len(true) else 0.0),
        "reliability_table": table,
    }


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


def letter_counts(texts: Sequence[str]) -> np.ndarray:
    return np.asarray([lexical_letter_count(text) for text in texts], dtype=np.int64)


def top_two(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-1/top-2 indices, top-1 score and the policy margin."""

    order = ranked(proba)
    rows = np.arange(len(proba))
    first, second = order[:, 0], order[:, 1]
    top1 = proba[rows, first]
    margin = np.clip(top1 - proba[rows, second], 0.0, 1.0)
    return first, second, top1, margin


def vectorized_acceptance(
    proba: np.ndarray,
    letters: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    minimum_letter_count: int,
) -> np.ndarray:
    """Vectorized equivalent of ``ReliabilityPolicy.evaluate`` (True = accept).

    ``policy_decisions`` applies the real policy object row by row; the test
    suite asserts that both agree exactly.
    """

    _, _, top1, margin = top_two(proba)
    return (
        (letters >= minimum_letter_count)
        & ~(top1 < confidence_threshold)
        & ~(margin < margin_threshold)
    )


def policy_decisions(
    policy: ReliabilityPolicy,
    texts: Sequence[str],
    proba: np.ndarray,
    labels: Sequence[str],
) -> list[tuple[str, str | None]]:
    """Apply the project's ``ReliabilityPolicy`` to each row: (language, reason)."""

    order = ranked(proba)
    decisions = []
    for text, row, idx in zip(texts, proba, order, strict=True):
        predictions = tuple(
            Prediction(LanguageCode(labels[j]), float(row[j])) for j in idx
        )
        decision = policy.evaluate(text, predictions)
        decisions.append(
            (
                decision.language.value,
                decision.reason.value if decision.reason else None,
            )
        )
    return decisions


def selective_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    accepted: np.ndarray,
    labels: Sequence[str],
    reasons: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Coverage, abstention rate and selective metrics for an accept mask."""

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    accepted = np.asarray(accepted, dtype=bool)
    n = len(labels)
    total = len(true)
    correct = pred == true
    covered = int(accepted.sum())
    matrix = np.zeros((n, n + 1), dtype=np.int64)
    np.add.at(matrix, (true, np.where(accepted, pred, n)), 1)
    per_class = {}
    for i, label in enumerate(labels):
        members = true == i
        support = int(members.sum())
        kept = int((members & accepted).sum())
        right = int((members & accepted & correct).sum())
        per_class[label] = {
            "support": support,
            "coverage": rnd(kept / support if support else None),
            "selective_recall": rnd(right / kept if kept else None),
            "coverage_adjusted_recall": rnd(right / support if support else None),
        }
    result = {
        "records": total,
        "accepted": covered,
        "coverage": rnd(covered / total if total else 0.0),
        "abstention_rate": rnd(1 - covered / total if total else 0.0),
        "selective_accuracy": rnd(
            (correct & accepted).sum() / covered if covered else None
        ),
        "coverage_adjusted_accuracy": rnd(
            (correct & accepted).sum() / total if total else 0.0
        ),
        "selective_macro_f1": rnd(
            macro_f1(true[accepted], pred[accepted], n) if covered else None
        ),
        "per_class": per_class,
        "confusion_with_und": {
            "labels": list(labels),
            "columns": [*labels, LanguageCode.UND.value],
            "matrix": matrix.tolist(),
        },
    }
    if reasons is not None:
        counts: dict[str, int] = {}
        for reason in reasons:
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
        result["abstention_reasons"] = dict(sorted(counts.items()))
    return result


def risk_coverage(
    proba: np.ndarray, y_true: Sequence[int], points: int = 101
) -> dict[str, Any]:
    """Selective risk as top-1 confidence thresholds sweep from strict to none.

    Records are ranked by top-1 score (ties by index for determinism). AURC is
    the mean selective risk over all N prefix coverages.
    """

    true = np.asarray(y_true, dtype=np.int64)
    total = len(true)
    if not total:
        return {"aurc": None, "curve": []}
    confidence = proba.max(axis=1)
    errors = (proba.argmax(axis=1) != true).astype(np.float64)
    order = np.lexsort((np.arange(total), -confidence))
    cumulative = np.cumsum(errors[order])
    counts = np.arange(1, total + 1)
    risk = cumulative / counts
    curve = []
    for coverage in np.linspace(0, 1, points)[1:]:
        k = max(1, round(coverage * total))
        curve.append(
            {
                "coverage": rnd(k / total),
                "risk": rnd(risk[k - 1]),
                "selective_accuracy": rnd(1 - risk[k - 1]),
                "threshold": rnd(confidence[order][k - 1]),
            }
        )
    return {"aurc": rnd(risk.mean()), "curve": curve}


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def wilson_interval(
    successes: int, total: int, confidence: float
) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = _normal_quantile(0.5 + confidence / 2)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [rnd(max(0.0, centre - half)), rnd(min(1.0, centre + half))]


def _normal_quantile(p: float) -> float:
    """Acklam's rational approximation (|error| < 1.2e-9)."""

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]  # fmt: skip
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    low = 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        return num / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - low:
        return -_normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    return num / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def group_bootstrap(
    y_true: Sequence[int],
    predictions: dict[str, Sequence[int]],
    groups: Sequence[str],
    n_labels: int,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Cluster bootstrap of macro-F1 (resampling document groups).

    Each model's macro-F1 interval and every pairwise difference are computed
    on the same resampled groups (a paired comparison). Groups, not records,
    are resampled because records in one translation group are dependent.
    """

    true = np.asarray(y_true, dtype=np.int64)
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(group_index.max()) + 1 if len(group_index) else 0
    names = sorted(predictions)
    preds = {name: np.asarray(predictions[name], dtype=np.int64) for name in names}
    # Per-group confusion counts make every resample a weighted sum.
    cells = n_labels * n_labels
    per_group = {name: np.zeros((n_groups, cells), dtype=np.int64) for name in names}
    for name in names:
        np.add.at(per_group[name], (group_index, true * n_labels + preds[name]), 1)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(resamples) for name in names}
    for r in range(resamples):
        weights = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        for name in names:
            matrix = (weights @ per_group[name]).reshape(n_labels, n_labels)
            _, _, f1, support = _prf_from_confusion(matrix)
            samples[name][r] = f1[support > 0].mean()
    alpha = (1 - confidence) / 2

    def interval(values: np.ndarray) -> list[float | None]:
        return [rnd(np.quantile(values, alpha)), rnd(np.quantile(values, 1 - alpha))]

    result: dict[str, Any] = {
        "unit": "document_id",
        "groups": n_groups,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "macro_f1": {
            name: {
                "point": rnd(macro_f1(true, preds[name], n_labels)),
                "interval": interval(samples[name]),
            }
            for name in names
        },
        "differences": {},
    }
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            diff = samples[left] - samples[right]
            result["differences"][f"{left}-{right}"] = {
                "point": rnd(
                    macro_f1(true, preds[left], n_labels)
                    - macro_f1(true, preds[right], n_labels)
                ),
                "interval": interval(diff),
                "share_of_resamples_left_better": rnd((diff > 0).mean()),
            }
    return result

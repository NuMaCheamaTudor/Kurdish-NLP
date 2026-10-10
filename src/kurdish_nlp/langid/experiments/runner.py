"""Experiment orchestration and protocol enforcement.

Phases (``kurdish-langid-experiment <phase> --config ...``):

``preflight``  verify Dataset V1, the UD benchmark and the environment.
``select``     train every candidate on train, select on dev_tune, calibrate and
               choose thresholds on dev_calib, then write ``selection.lock.json``.
               Never reads test or UD predictions.
``evaluate``   internal test, only with a valid lock whose artifact hashes match.
``external``   UD strict and exploratory slices, separately, with the same lock.
``report``     comparison tables, slices, error analysis and figures.
``reproduce``  reload/retrain checks and report-regeneration determinism.
``all``        every phase in order.

After ``evaluate`` has written ``test_evaluation.json``, ``select`` refuses to
run: changing models after seeing test results would be test-set tuning.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kurdish_nlp.langid.build.export import write_json, write_jsonl
from kurdish_nlp.langid.build.grouping import stable_key
from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.experiments.calibration import (
    apply_temperature,
    calibrate,
    select_thresholds,
)
from kurdish_nlp.langid.experiments.config import (
    ExperimentConfig,
    load_experiment_config,
)
from kurdish_nlp.langid.experiments.data import (
    ExampleSet,
    LoadedData,
    load_experiment_data,
    load_external_benchmark,
    write_fasttext_training_file,
)
from kurdish_nlp.langid.experiments.detector import (
    CalibratedLanguageDetector,
    load_classifier,
    write_inference_spec,
)
from kurdish_nlp.langid.experiments.evaluation import (
    classification_report,
    letter_counts,
    log_loss,
    policy_decisions,
    probability_report,
    risk_coverage,
    rnd,
    selective_report,
    vectorized_acceptance,
    wilson_interval,
)
from kurdish_nlp.langid.experiments.slices import predicted_distribution, slice_rows
from kurdish_nlp.langid.reliability import ReliabilityPolicy

FAMILIES = ("fasttext", "tfidf_lr")
DISPLAY = {"fasttext": "fastText", "tfidf_lr": "TF-IDF + LR"}
INTERNAL_ROLES = ("dev_tune", "dev_calib", "test")
LOCK_FILE = "selection.lock.json"
TEST_MARKER = "test_evaluation.json"
Progress = Callable[[str], None]


class ProtocolError(RuntimeError):
    """Raised when a phase would violate the train/dev/test protocol."""


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


class Paths:
    def __init__(self, config: ExperimentConfig) -> None:
        self.root = config.artifact_root
        self.data = self.root / "data"
        self.reports = self.root / "reports"
        self.lock = self.root / LOCK_FILE
        self.test_marker = self.root / TEST_MARKER
        self.manifest = self.root / "experiment_manifest.json"
        self.reproducibility = self.root / "reproducibility.json"
        self.preflight = self.root / "preflight.json"

    def model_dir(self, family: str) -> Path:
        return self.root / family

    def predictions(self, family: str, role: str) -> Path:
        return self.model_dir(family) / "predictions" / f"{role}.jsonl"


def _artifact_name(config: ExperimentConfig, family: str) -> str:
    return config.fasttext.artifact if family == "fasttext" else config.tfidf.artifact


# --------------------------------------------------------------------------
# Prediction files
# --------------------------------------------------------------------------


def save_predictions(
    path: Path, examples: ExampleSet, proba: np.ndarray, config: ExperimentConfig
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        {"id": e.id, "label": e.label.value, "proba": [float(v) for v in row]}
        for e, row in zip(examples.examples, proba, strict=True)
    )
    write_jsonl(path, rows)
    return file_sha256(path)


def read_predictions(path: Path, examples: ExampleSet) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"predictions not found: {path}; run the earlier phase")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row["id"] for row in rows] != examples.ids:
        raise ProtocolError(f"{path} does not match the {examples.name} examples")
    return np.asarray([row["proba"] for row in rows], dtype=np.float64)


def argmax(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1).astype(np.int64)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _command(args: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            list(args), capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info(root: Path) -> dict[str, Any]:
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    cpu = _command(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    probe = root if root.exists() else Path.cwd()
    usage = shutil.disk_usage(probe)
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory,
        "disk_free_bytes": usage.free,
    }


def library_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
        "fasttext",
        "matplotlib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_state() -> dict[str, Any]:
    commit = _command(["git", "rev-parse", "HEAD"])
    status = _command(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty_worktree": bool(status) if status is not None else None,
    }


def _thread_limit(config: ExperimentConfig):
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=config.resources.max_threads)


# --------------------------------------------------------------------------
# Phase A: preflight
# --------------------------------------------------------------------------


def run_preflight(config: ExperimentConfig, progress: Progress = _stderr) -> dict:
    paths = Paths(config)
    progress("[preflight] verifying Dataset V1 hashes, records and isolation")
    data = load_experiment_data(config)
    progress("[preflight] verifying the UD benchmark and its separation")
    training_texts = [
        e.text
        for s in (data.train, data.dev_tune, data.dev_calib, data.test)
        for e in s.examples
    ]
    external = load_external_benchmark(config, training_texts)
    fasttext_api = _inspect_fasttext()
    hardware = hardware_info(paths.root)
    result = {
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "dataset": data.verification,
        "development_roles": data.role_report,
        "contributors": data.contributor_report,
        "external_benchmark": external.report,
        "fasttext_backend": fasttext_api,
        "libraries": library_versions(),
        "hardware": hardware,
        "resource_limits": {
            "max_threads": config.resources.max_threads,
            "memory_budget_bytes": config.resources.memory_budget_bytes,
            "memory_budget_ok": hardware["memory_bytes"] is None
            or config.resources.memory_budget_bytes < hardware["memory_bytes"],
        },
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    write_json(paths.preflight, result)
    progress(f"[preflight] ok -> {paths.preflight}")
    return result


def _inspect_fasttext() -> dict[str, Any]:
    try:
        import fasttext  # type: ignore[import-not-found]
    except ImportError:
        return {"available": False}
    import importlib.metadata
    import inspect

    from kurdish_nlp.langid import fasttext_detector
    from kurdish_nlp.langid.experiments.fasttext_baseline import (
        UNINITIALIZED_MATRIX_VERSIONS,
    )

    source = inspect.getsource(fasttext.FastText._FastText.predict)
    version = importlib.metadata.version("fasttext")
    return {
        "available": True,
        "version": version,
        "training_safe_with_thread_1": version not in UNINITIALIZED_MATRIX_VERSIONS,
        "single_text_predict_uses_copy_false": "copy=False" in source,
        "project_adapter": "FastTextLanguageDetector wraps real models in "
        "NumpyCompatibleFastTextModel (list predict path, 1e-5 offset removed)",
        "probability_offset": fasttext_detector.FASTTEXT_PROBABILITY_OFFSET,
    }


# --------------------------------------------------------------------------
# Phases B-D: search, selection, calibration, freeze
# --------------------------------------------------------------------------


def _selection_key(row: dict[str, Any]) -> tuple:
    return (
        -row["dev_tune"]["macro_f1"],
        row["dev_tune"]["log_loss"],
        row["model_bytes"],
    )


def _tune_metrics(
    config: ExperimentConfig, proba: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    report = classification_report(y, argmax(proba), config.label_values)
    return {
        "macro_f1": report["macro_f1"],
        "accuracy": report["accuracy"],
        "log_loss": rnd(log_loss(proba, y, config.calibration.probability_floor)),
        "focus_f1": {
            label.value: report["per_class"][label.value]["f1"]
            for label in config.focus_labels
        },
    }


def search_fasttext(
    config: ExperimentConfig, data: LoadedData, train_file: Path, progress: Progress
):
    from kurdish_nlp.langid.experiments.fasttext_baseline import (
        FastTextModel,
        memory_estimate,
    )

    labels = config.label_values
    y_tune = np.asarray(data.dev_tune.label_indices(config))
    vocabulary = len({token for e in data.train.examples for token in e.text.split()})
    candidates = config.fasttext.candidates()
    log, best, best_row = [], None, None
    scratch = Path(tempfile.mkdtemp(prefix="ft-candidate-", dir=config.artifact_root))
    try:
        for index, params in enumerate(candidates):
            row: dict[str, Any] = {"candidate": index, "params": params}
            estimate = memory_estimate(params, vocabulary)
            row["memory_estimate_bytes"] = estimate
            if estimate > config.resources.memory_budget_bytes:
                row["status"] = "skipped_memory_budget"
                log.append(row)
                continue
            start = time.perf_counter()
            model = FastTextModel.train(
                train_file, params, seed=config.seed, labels=labels
            )
            row["fit_seconds"] = round(time.perf_counter() - start, 3)
            proba = model.predict_proba(data.dev_tune.texts)
            row["dev_tune"] = _tune_metrics(config, proba, y_tune)
            probe = scratch / "candidate.bin"
            model.model.save_model(str(probe))
            row["model_bytes"] = probe.stat().st_size
            probe.unlink()
            row["vocabulary_size"] = model.vocabulary_size
            row["status"] = "ok"
            log.append(row)
            progress(
                f"[select] fastText {index + 1}/{len(candidates)} "
                f"macroF1={row['dev_tune']['macro_f1']:.4f} ({row['fit_seconds']}s) {params}"
            )
            if best_row is None or _selection_key(row) < _selection_key(best_row):
                best, best_row = model, row
            else:
                del model
            gc.collect()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if best is None:
        raise RuntimeError("no fastText candidate fitted within the resource budget")
    return best, best_row, log


def search_tfidf(config: ExperimentConfig, data: LoadedData, progress: Progress):
    from kurdish_nlp.langid.experiments.tfidf_baseline import (
        TfidfLogisticModel,
        build_classifier,
        build_vectorizer,
        csr_bytes,
        fit_classifier,
        lbfgs_memory_estimate,
    )

    spec = config.tfidf
    labels = config.label_values
    y_train = np.asarray(data.train.label_indices(config))
    y_tune = np.asarray(data.dev_tune.label_indices(config))
    vectorizers = spec.vectorizer_candidates()
    classifiers = spec.classifier_candidates()
    total = len(vectorizers) * len(classifiers)
    log, best, best_row, index = [], None, None, 0
    for vparams in vectorizers:
        start = time.perf_counter()
        vectorizer = build_vectorizer(spec.vectorizer_fixed, vparams)
        x_train = vectorizer.fit_transform(data.train.texts)
        vectorize_seconds = round(time.perf_counter() - start, 3)
        x_tune = vectorizer.transform(data.dev_tune.texts)
        n_features = len(vectorizer.vocabulary_)
        vocab_bytes = sum(len(t.encode()) + 64 for t in vectorizer.vocabulary_)
        estimate = csr_bytes(x_train) + lbfgs_memory_estimate(n_features, len(labels))
        for cparams in classifiers:
            row: dict[str, Any] = {
                "candidate": index,
                "vectorizer": {**vparams, "ngram_range": list(vparams["ngram_range"])},
                "classifier": cparams,
                "n_features": n_features,
                "train_nnz": int(x_train.nnz),
                "vectorize_seconds": vectorize_seconds,
                "memory_estimate_bytes": estimate,
            }
            index += 1
            if estimate > config.resources.memory_budget_bytes:
                row["status"] = "skipped_memory_budget"
                log.append(row)
                continue
            classifier = build_classifier(spec.classifier_fixed, cparams, config.seed)
            row.update(fit_classifier(classifier, x_train, y_train))
            model = TfidfLogisticModel.from_parts(vectorizer, classifier, labels)
            raw = classifier.predict_proba(x_tune)
            proba = model.align(raw)
            row["dev_tune"] = _tune_metrics(config, proba, y_tune)
            row["model_bytes"] = int(n_features * (len(labels) + 1) * 8 + vocab_bytes)
            row["model_bytes_is_estimate"] = True
            row["status"] = "ok"
            log.append(row)
            progress(
                f"[select] TF-IDF {index}/{total} macroF1={row['dev_tune']['macro_f1']:.4f} "
                f"({row['fit_seconds']}s, {row['iterations']} it) {vparams} {cparams}"
            )
            if best_row is None or _selection_key(row) < _selection_key(best_row):
                best, best_row = model, row
        del x_train, x_tune
        gc.collect()
    if best is None:
        raise RuntimeError("no TF-IDF candidate fitted within the resource budget")
    return best, best_row, log


def run_select(
    config: ExperimentConfig, *, force: bool = False, progress: Progress = _stderr
) -> dict[str, Any]:
    paths = Paths(config)
    if paths.test_marker.exists():
        raise ProtocolError(
            "the internal test set has already been evaluated with the frozen "
            "selection; re-running selection now would be test-set tuning"
        )
    if paths.lock.exists() and not force:
        raise ProtocolError(
            f"{paths.lock} already exists; pass --force to redo selection "
            "(allowed only before test evaluation)"
        )
    paths.root.mkdir(parents=True, exist_ok=True)
    data = load_experiment_data(config)
    paths.data.mkdir(parents=True, exist_ok=True)
    roles = {
        "report": data.role_report,
        "records": {
            **{e.id: "dev_tune" for e in data.dev_tune.examples},
            **{
                e.id: f"dev_calib:{fold}"
                for e, fold in zip(
                    data.dev_calib.examples, data.calibration_folds, strict=True
                )
            },
        },
    }
    write_json(paths.data / "dev_roles.json", roles)
    train_file = write_fasttext_training_file(
        data.train, paths.data / "train.fasttext.txt"
    )
    progress(f"[select] fastText training file: {train_file['lines']} lines")

    lock_families: dict[str, Any] = {}
    with _thread_limit(config):
        searches = {}
        start = time.perf_counter()
        searches["fasttext"] = search_fasttext(
            config, data, Path(train_file["path"]), progress
        ) + (round(time.perf_counter() - start, 3),)
        start = time.perf_counter()
        searches["tfidf_lr"] = search_tfidf(config, data, progress) + (
            round(time.perf_counter() - start, 3),
        )
        for family in FAMILIES:
            model, best_row, log, search_seconds = searches[family]
            lock_families[family] = _freeze_family(
                config,
                paths,
                data,
                family,
                model,
                best_row,
                log,
                search_seconds,
                train_file,
                progress,
            )
            del model
            searches[family] = None
            gc.collect()

    lock = {
        "experiment_id": config.experiment_id,
        "experiment_version": config.experiment_version,
        "config_sha256": config.sha256,
        "dataset_files_sha256": data.verification["files_sha256"],
        "development_assignment_sha256": data.role_report["assignment_sha256"],
        "fasttext_training_file_sha256": train_file["sha256"],
        "families": lock_families,
        "protocol": {
            "selection_role": config.selection_role,
            "selection_metric": config.selection_metric,
            "calibration_role": config.calibration.fit_role,
            "threshold_role": config.abstention.fit_role,
            "test_used_for_selection": False,
            "external_benchmark_used_for_selection": False,
        },
    }
    write_json(paths.lock, lock)
    write_manifest(config)
    progress(f"[select] selection frozen -> {paths.lock}")
    return lock


def _freeze_family(
    config: ExperimentConfig,
    paths: Paths,
    data: LoadedData,
    family: str,
    model: Any,
    best_row: dict[str, Any],
    log: list[dict[str, Any]],
    search_seconds: float,
    train_file: dict[str, Any],
    progress: Progress,
) -> dict[str, Any]:
    labels = config.label_values
    directory = paths.model_dir(family)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / _artifact_name(config, family)
    meta = model.save(artifact)

    predictions = {}
    for role in ("dev_tune", "dev_calib"):
        examples = data.role(role)
        proba = model.predict_proba(examples.texts)
        predictions[role] = save_predictions(
            paths.predictions(family, role), examples, proba, config
        )
        if role == "dev_calib":
            calib_proba = proba
    y_calib = np.asarray(data.dev_calib.label_indices(config))
    calibration = calibrate(
        calib_proba, y_calib, data.calibration_folds, config.calibration
    )
    temperature = float(calibration["temperature"])
    scores = (
        apply_temperature(
            calib_proba, temperature, config.calibration.probability_floor
        )
        if temperature != 1.0
        else calib_proba
    )
    thresholds = select_thresholds(
        scores, y_calib, letter_counts(data.dev_calib.texts), config.abstention
    )
    policy = {
        "confidence_threshold": thresholds["confidence_threshold"],
        "margin_threshold": thresholds["margin_threshold"],
        "minimum_letter_count": config.abstention.minimum_letter_count,
    }
    reference = {
        "confidence_threshold": config.abstention.reference_confidence,
        "margin_threshold": config.abstention.reference_margin,
        "minimum_letter_count": config.abstention.minimum_letter_count,
    }
    selected = (
        best_row["params"]
        if family == "fasttext"
        else {
            "vectorizer": {**config.tfidf.vectorizer_fixed, **best_row["vectorizer"]},
            "classifier": {**config.tfidf.classifier_fixed, **best_row["classifier"]},
        }
    )
    training = {
        "family": family,
        "selected_candidate": best_row["candidate"],
        "selected_params": selected,
        "seed": config.seed,
        "selection": {
            "role": config.selection_role,
            "metric": config.selection_metric,
            "tie_breakers": list(config.selection_tie_breakers),
            "dev_tune": best_row["dev_tune"],
            "note": "dev_tune scores are optimistic: they were used to choose this candidate",
        },
        "search": {
            "candidates": len(log),
            "fitted": sum(row["status"] == "ok" for row in log),
            "log": [
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"fit_seconds", "vectorize_seconds"}
                }
                for row in log
            ],
        },
        "training_data": {
            "role": "train",
            "records": len(data.train),
            "fasttext_file_sha256": train_file["sha256"]
            if family == "fasttext"
            else None,
        },
        "artifact": meta,
        "predictions_sha256": predictions,
    }
    write_json(directory / "training.json", training)
    write_json(
        directory / "calibration.json",
        {"family": family, "calibration": calibration, "abstention": thresholds},
    )
    write_json(
        directory / "runtime.json",
        {
            "family": family,
            "search_seconds": search_seconds,
            "candidate_fit_seconds": [
                {
                    "candidate": row["candidate"],
                    "fit_seconds": row.get("fit_seconds"),
                    "vectorize_seconds": row.get("vectorize_seconds"),
                }
                for row in log
            ],
            "selected_fit_seconds": best_row.get("fit_seconds"),
            "selected_vectorize_seconds": best_row.get("vectorize_seconds"),
            "peak_rss_bytes_after_select": peak_rss_bytes(),
        },
    )
    write_inference_spec(
        directory,
        {
            "family": family,
            "artifact": artifact.name,
            "artifact_sha256": meta["sha256"],
            "labels": list(labels),
            "temperature": temperature,
            "probability_floor": config.calibration.probability_floor,
            "policy": policy,
            "reference_policy": reference,
            "model_identifier": f"kurdish-langid-{config.experiment_id}-{family}",
            "model_version": config.experiment_version,
            "score_semantics": "calibrated score estimates; not guaranteed "
            "probabilities of correctness",
        },
    )
    progress(
        f"[select] {DISPLAY[family]} frozen: T={temperature:.4f} "
        f"thresholds conf={policy['confidence_threshold']} margin={policy['margin_threshold']}"
    )
    return {
        "artifact": artifact.name,
        "artifact_sha256": meta["sha256"],
        "artifact_bytes": meta["bytes"],
        "selected_params": selected,
        "dev_tune_macro_f1": best_row["dev_tune"]["macro_f1"],
        "temperature": temperature,
        "calibration_adopted": calibration["adopted"],
        "policy": policy,
        "reference_policy": reference,
        "inference_spec_sha256": file_sha256(directory / "inference.json"),
        "dev_predictions_sha256": predictions,
    }


# --------------------------------------------------------------------------
# Lock verification
# --------------------------------------------------------------------------


def verify_lock(config: ExperimentConfig) -> dict[str, Any]:
    paths = Paths(config)
    if not paths.lock.is_file():
        raise ProtocolError(
            f"no selection lock at {paths.lock}: run 'select' before any test or "
            "external evaluation"
        )
    lock = json.loads(paths.lock.read_text(encoding="utf-8"))
    if lock["config_sha256"] != config.sha256:
        raise ProtocolError(
            "the experiment configuration changed after selection was frozen"
        )
    for family, entry in lock["families"].items():
        artifact = paths.model_dir(family) / entry["artifact"]
        if not artifact.is_file():
            raise FileNotFoundError(f"frozen model artifact is missing: {artifact}")
        actual = file_sha256(artifact)
        if actual != entry["artifact_sha256"]:
            raise ProtocolError(f"{artifact} changed after selection ({actual})")
        spec = paths.model_dir(family) / "inference.json"
        if file_sha256(spec) != entry["inference_spec_sha256"]:
            raise ProtocolError(f"{spec} changed after selection")
    lock["lock_sha256"] = file_sha256(paths.lock)
    return lock


# --------------------------------------------------------------------------
# Shared role evaluation
# --------------------------------------------------------------------------


def evaluate_role(
    config: ExperimentConfig,
    examples: ExampleSet,
    raw: np.ndarray,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    labels = config.label_values
    y = np.asarray(examples.label_indices(config))
    pred = argmax(raw)
    temperature = float(frozen["temperature"])
    floor = config.calibration.probability_floor
    calibrated = (
        apply_temperature(raw, temperature, floor) if temperature != 1.0 else raw
    )
    letters = letter_counts(examples.texts)
    abstention = {"none": selective_report(y, pred, np.ones(len(y), bool), labels)}
    for name, scores, policy in (
        ("reference_policy_raw", raw, frozen["reference_policy"]),
        ("tuned_policy_calibrated", calibrated, frozen["policy"]),
    ):
        decisions = policy_decisions(
            ReliabilityPolicy(**policy), examples.texts, scores, labels
        )
        accepted = np.asarray([language != "und" for language, _ in decisions])
        expected = vectorized_acceptance(scores, letters, **policy)
        if not np.array_equal(accepted, expected):
            raise AssertionError(
                "vectorized abstention disagrees with ReliabilityPolicy"
            )
        abstention[name] = selective_report(
            y, pred, accepted, labels, [reason for _, reason in decisions]
        )
        abstention[name]["policy"] = policy
    return {
        "role": examples.name,
        "records": len(examples),
        "classification": classification_report(y, pred, labels),
        "probabilities": {
            "raw": probability_report(
                raw, y, floor=floor, ece_bins=config.calibration.ece_bins
            ),
            "calibrated": probability_report(
                calibrated, y, floor=floor, ece_bins=config.calibration.ece_bins
            ),
            "temperature": temperature,
        },
        "abstention": abstention,
        "risk_coverage": {
            "raw": risk_coverage(raw, y),
            "calibrated": risk_coverage(calibrated, y),
        },
    }


def _latency(
    config: ExperimentConfig, detector: CalibratedLanguageDetector, test: ExampleSet
) -> dict[str, Any]:
    ordered = sorted(
        test.examples, key=lambda e: stable_key(config.seed, "latency", e.id)
    )
    sample = [e.text for e in ordered[: config.evaluation.latency_samples]]
    for text in sample[: config.evaluation.latency_warmup]:
        detector.detect(text)
    timings = []
    for text in sample:
        start = time.perf_counter()
        detector.detect(text)
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    detector.predict_proba_batch(test.texts)
    batch_seconds = time.perf_counter() - start
    values = np.asarray(timings)
    return {
        "single_text_ms": {
            "samples": len(values),
            "mean": rnd(values.mean()),
            "median": rnd(np.median(values)),
            "p95": rnd(np.quantile(values, 0.95)),
        },
        "batch": {
            "records": len(test),
            "seconds": rnd(batch_seconds),
            "texts_per_second": rnd(len(test) / batch_seconds),
        },
        "threads": config.resources.max_threads,
    }


def _detector(config: ExperimentConfig, family: str, classifier: Any, frozen: dict):
    return CalibratedLanguageDetector(
        classifier,
        temperature=frozen["temperature"],
        policy=ReliabilityPolicy(**frozen["policy"]),
        probability_floor=config.calibration.probability_floor,
        model_identifier=f"kurdish-langid-{config.experiment_id}-{family}",
        model_version=config.experiment_version,
    )


# --------------------------------------------------------------------------
# Phase E: internal test
# --------------------------------------------------------------------------


def run_evaluate(config: ExperimentConfig, progress: Progress = _stderr) -> dict:
    lock = verify_lock(config)
    paths = Paths(config)
    data = load_experiment_data(config)
    marker: dict[str, Any] = {"lock_sha256": lock["lock_sha256"], "families": {}}
    with _thread_limit(config):
        for family in FAMILIES:
            frozen = lock["families"][family]
            artifact = paths.model_dir(family) / frozen["artifact"]
            classifier = load_classifier(family, artifact, config.label_values)
            reload_check = {}
            for role in ("dev_tune", "dev_calib"):
                stored = read_predictions(
                    paths.predictions(family, role), data.role(role)
                )
                fresh = classifier.predict_proba(data.role(role).texts)
                reload_check[role] = {
                    "max_abs_difference": float(np.abs(stored - fresh).max()),
                    "argmax_identical": bool(
                        np.array_equal(argmax(stored), argmax(fresh))
                    ),
                }
            raw_test = classifier.predict_proba(data.test.texts)
            test_hash = save_predictions(
                paths.predictions(family, "test"), data.test, raw_test, config
            )
            evaluation = {
                "family": family,
                "lock_sha256": lock["lock_sha256"],
                "reload_check_vs_selection": reload_check,
                "roles": {
                    "dev_tune": evaluate_role(
                        config,
                        data.dev_tune,
                        read_predictions(
                            paths.predictions(family, "dev_tune"), data.dev_tune
                        ),
                        frozen,
                    ),
                    "dev_calib": evaluate_role(
                        config,
                        data.dev_calib,
                        read_predictions(
                            paths.predictions(family, "dev_calib"), data.dev_calib
                        ),
                        frozen,
                    ),
                    "test": evaluate_role(config, data.test, raw_test, frozen),
                },
                "role_notes": {
                    "dev_tune": "used for hyperparameter selection (optimistic)",
                    "dev_calib": "used to fit temperature and thresholds (in-sample for those)",
                    "test": "untouched until the selection lock; the honest internal estimate",
                },
            }
            write_json(paths.model_dir(family) / "evaluation.json", evaluation)
            runtime_path = paths.model_dir(family) / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["inference"] = _latency(
                config, _detector(config, family, classifier, frozen), data.test
            )
            runtime["peak_rss_bytes_after_evaluate"] = peak_rss_bytes()
            write_json(runtime_path, runtime)
            marker["families"][family] = {
                "evaluation_sha256": file_sha256(
                    paths.model_dir(family) / "evaluation.json"
                ),
                "test_predictions_sha256": test_hash,
                "test_macro_f1": evaluation["roles"]["test"]["classification"][
                    "macro_f1"
                ],
            }
            progress(
                f"[evaluate] {DISPLAY[family]} test macro-F1 "
                f"{marker['families'][family]['test_macro_f1']:.4f}"
            )
            del classifier
            gc.collect()
    write_json(paths.test_marker, marker)
    write_manifest(config)
    return marker


# --------------------------------------------------------------------------
# Phase F: external benchmark
# --------------------------------------------------------------------------


def external_slice_report(
    config: ExperimentConfig,
    examples: ExampleSet,
    raw: np.ndarray,
    frozen: dict[str, Any],
    *,
    title: str,
    exploratory: bool,
) -> dict[str, Any]:
    labels = config.label_values
    y = np.asarray(examples.label_indices(config))
    target = labels[int(y[0])]
    pred = argmax(raw)
    temperature = float(frozen["temperature"])
    floor = config.calibration.probability_floor
    calibrated = (
        apply_temperature(raw, temperature, floor) if temperature != 1.0 else raw
    )
    correct = pred == y
    confidence = 0.95
    result: dict[str, Any] = {
        "title": title,
        "status": "EXPLORATORY — prior repository exposure"
        if exploratory
        else "STRICT",
        "expected_language": target,
        "records": len(examples),
        "note": "single-label slice: recall of the expected label and the confusion "
        "distribution are reported; seven-class macro-F1 is undefined here",
        "argmax": {
            "correct": int(correct.sum()),
            "recall": rnd(correct.mean()),
            "recall_wilson_95": wilson_interval(int(correct.sum()), len(y), confidence),
            "prediction_distribution": predicted_distribution(pred, labels),
            "mean_top1_calibrated_confidence": rnd(calibrated.max(axis=1).mean()),
            "mean_calibrated_score_for_expected_label": rnd(
                calibrated[np.arange(len(y)), y].mean()
            ),
        },
    }
    for name, scores, policy in (
        ("tuned_policy_calibrated", calibrated, frozen["policy"]),
        ("reference_policy_raw", raw, frozen["reference_policy"]),
    ):
        decisions = policy_decisions(
            ReliabilityPolicy(**policy), examples.texts, scores, labels
        )
        accepted = np.asarray([language != "und" for language, _ in decisions])
        reasons: dict[str, int] = {}
        for _, reason in decisions:
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        right = int((correct & accepted).sum())
        result[name] = {
            "policy": policy,
            "accepted": int(accepted.sum()),
            "abstention_rate": rnd(1 - accepted.mean()),
            "selective_recall": rnd(right / accepted.sum() if accepted.any() else None),
            "coverage_adjusted_recall": rnd(right / len(y)),
            "coverage_adjusted_recall_wilson_95": wilson_interval(
                right, len(y), confidence
            ),
            "prediction_distribution_with_und": predicted_distribution(
                pred, labels, accepted
            ),
            "abstention_reasons": dict(sorted(reasons.items())),
        }
    accepted = np.asarray(
        [
            language != "und"
            for language, _ in policy_decisions(
                ReliabilityPolicy(**frozen["policy"]),
                examples.texts,
                calibrated,
                labels,
            )
        ]
    )
    for dimension in ("length", "script", "domain"):
        result[f"by_{dimension}"] = [
            {
                **row,
                "recall_wilson_95": wilson_interval(
                    row["correct"], row["support"], confidence
                ),
            }
            for row in slice_rows(
                examples.examples,
                y,
                pred,
                labels,
                dimension,
                by_language=False,
                min_support=config.evaluation.min_slice_support,
                accepted=accepted,
            )
        ]
    return result


def run_external(config: ExperimentConfig, progress: Progress = _stderr) -> dict:
    lock = verify_lock(config)
    paths = Paths(config)
    data = load_experiment_data(config)
    texts = [
        e.text
        for s in (data.train, data.dev_tune, data.dev_calib, data.test)
        for e in s.examples
    ]
    external = load_external_benchmark(config, texts)
    spec = config.external
    combined: dict[str, Any] = {
        "benchmark": external.report,
        "lock_sha256": lock["lock_sha256"],
        "protocol": "models, temperatures and thresholds were frozen before this "
        "benchmark was scored; UD was never used for tuning, calibration or selection",
        "strict": {},
        "exploratory": {},
    }
    with _thread_limit(config):
        for family in FAMILIES:
            frozen = lock["families"][family]
            artifact = paths.model_dir(family) / frozen["artifact"]
            classifier = load_classifier(family, artifact, config.label_values)
            family_result = {}
            for key, examples, slice_spec, exploratory in (
                ("strict", external.strict, spec.strict, False),
                ("exploratory", external.exploratory, spec.exploratory, True),
            ):
                raw = classifier.predict_proba(examples.texts)
                save_predictions(
                    paths.predictions(family, examples.name), examples, raw, config
                )
                report = external_slice_report(
                    config,
                    examples,
                    raw,
                    frozen,
                    title=slice_spec.title,
                    exploratory=exploratory,
                )
                family_result[key] = report
                combined[key][family] = report
            write_json(
                paths.model_dir(family) / "evaluation_external.json",
                {"family": family, "lock_sha256": lock["lock_sha256"], **family_result},
            )
            progress(
                f"[external] {DISPLAY[family]} strict sdh recall "
                f"{family_result['strict']['argmax']['recall']:.4f}; exploratory kmr recall "
                f"{family_result['exploratory']['argmax']['recall']:.4f}"
            )
            del classifier
            gc.collect()
    write_manifest(config)
    return combined


# --------------------------------------------------------------------------
# Phase G: reports
# --------------------------------------------------------------------------


def run_report(
    config: ExperimentConfig,
    progress: Progress = _stderr,
    *,
    output_dir: Path | None = None,
    data: LoadedData | None = None,
) -> dict[str, str]:
    from kurdish_nlp.langid.experiments.reporting import build_reports

    lock = verify_lock(config)
    paths = Paths(config)
    if not paths.test_marker.is_file():
        raise ProtocolError("run 'evaluate' before 'report'")
    data = data or load_experiment_data(config)
    hashes = build_reports(config, lock, data, output_dir or paths.reports)
    if output_dir is None:
        write_manifest(config)
    progress(f"[report] wrote {len(hashes)} report files")
    return hashes


# --------------------------------------------------------------------------
# Phase H: reproducibility
# --------------------------------------------------------------------------


def run_reproduce(config: ExperimentConfig, progress: Progress = _stderr) -> dict:
    lock = verify_lock(config)
    paths = Paths(config)
    data = load_experiment_data(config)
    result: dict[str, Any] = {"lock_sha256": lock["lock_sha256"], "families": {}}
    with _thread_limit(config):
        for family in FAMILIES:
            frozen = lock["families"][family]
            artifact = paths.model_dir(family) / frozen["artifact"]
            stored = read_predictions(paths.predictions(family, "test"), data.test)
            reloaded = load_classifier(family, artifact, config.label_values)
            fresh = reloaded.predict_proba(data.test.texts)
            entry: dict[str, Any] = {
                "reload": {
                    "max_abs_difference": float(np.abs(stored - fresh).max()),
                    "argmax_identical": bool(
                        np.array_equal(argmax(stored), argmax(fresh))
                    ),
                }
            }
            del reloaded
            start = time.perf_counter()
            retrained, retrained_artifact = _retrain(
                config, paths, data, family, frozen
            )
            entry["retrain_seconds"] = round(time.perf_counter() - start, 3)
            again = retrained.predict_proba(data.test.texts)
            retrained_hash = file_sha256(retrained_artifact)
            entry["retrain"] = {
                "artifact_sha256": retrained_hash,
                "artifact_bitwise_identical": retrained_hash
                == frozen["artifact_sha256"],
                "max_abs_difference": float(np.abs(stored - again).max()),
                "argmax_identical": bool(np.array_equal(argmax(stored), argmax(again))),
                "differing_predictions": int((argmax(stored) != argmax(again)).sum()),
            }
            shutil.rmtree(retrained_artifact.parent, ignore_errors=True)
            result["families"][family] = entry
            progress(f"[reproduce] {DISPLAY[family]}: {entry}")
            del retrained
            gc.collect()
    with tempfile.TemporaryDirectory(dir=paths.root) as tmp:
        regenerated = run_report(config, progress, output_dir=Path(tmp), data=data)
    official = {
        name: file_sha256(paths.reports / name)
        for name in regenerated
        if (paths.reports / name).is_file()
    }
    result["report_regeneration"] = {
        "files": len(regenerated),
        "identical": sorted(
            n for n in regenerated if official.get(n) == regenerated[n]
        ),
        "different": sorted(
            n for n in regenerated if official.get(n) != regenerated[n]
        ),
    }
    write_json(paths.reproducibility, result)
    write_manifest(config)
    return result


def _retrain(
    config: ExperimentConfig, paths: Paths, data: LoadedData, family: str, frozen: dict
):
    scratch = Path(tempfile.mkdtemp(prefix=f"retrain-{family}-", dir=paths.root))
    params = frozen["selected_params"]
    if family == "fasttext":
        from kurdish_nlp.langid.experiments.fasttext_baseline import FastTextModel

        train_file = write_fasttext_training_file(data.train, scratch / "train.txt")
        model = FastTextModel.train(
            train_file["path"], params, seed=config.seed, labels=config.label_values
        )
    else:
        from kurdish_nlp.langid.experiments.tfidf_baseline import TfidfLogisticModel

        vectorizer = dict(params["vectorizer"])
        model = TfidfLogisticModel.fit(
            data.train.texts,
            data.train.label_indices(config),
            vectorizer_fixed={},
            vectorizer_params=vectorizer,
            classifier_fixed={},
            classifier_params=params["classifier"],
            labels=config.label_values,
            seed=config.seed,
        )
    artifact = scratch / _artifact_name(config, family)
    model.save(artifact)
    return model, artifact


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _hash_tree(directory: Path, root: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def write_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Experiment manifest; timings live in runtime.json files, not in hashes."""

    paths = Paths(config)
    lock = json.loads(paths.lock.read_text()) if paths.lock.is_file() else None
    runtime = {}
    for family in FAMILIES:
        path = paths.model_dir(family) / "runtime.json"
        if path.is_file():
            runtime[family] = json.loads(path.read_text(encoding="utf-8"))
    outputs = {}
    for family in FAMILIES:
        for name in (
            "training.json",
            "calibration.json",
            "inference.json",
            "evaluation.json",
            "evaluation_external.json",
        ):
            path = paths.model_dir(family) / name
            if path.is_file():
                outputs[f"{family}/{name}"] = file_sha256(path)
        outputs.update(_hash_tree(paths.model_dir(family) / "predictions", paths.root))
    for path in (paths.lock, paths.test_marker, paths.preflight, paths.reproducibility):
        if path.is_file():
            outputs[path.name] = file_sha256(path)
    manifest = {
        "experiment_id": config.experiment_id,
        "experiment_version": config.experiment_version,
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "file_sha256": file_sha256(config.path),
        },
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "build_config_sha256": config.dataset.config_sha256,
            "files_sha256": dict(config.dataset.expected_sha256),
        },
        "external_benchmark": {
            "records_sha256": config.external.records.sha256,
            "manifest_sha256": config.external.manifest.sha256,
        },
        "seed": config.seed,
        "selected": {
            family: {
                "params": entry["selected_params"],
                "temperature": entry["temperature"],
                "policy": entry["policy"],
            }
            for family, entry in (lock or {}).get("families", {}).items()
        },
        "model_artifacts_sha256": {
            f"{family}/{entry['artifact']}": entry["artifact_sha256"]
            for family, entry in (lock or {}).get("families", {}).items()
        },
        "evaluation_outputs_sha256": outputs,
        "report_files_sha256": _hash_tree(paths.reports, paths.root),
        "libraries": library_versions(),
        "git": git_state(),
        "hardware": hardware_info(paths.root),
        "runtime": runtime,
        "known_limitations": KNOWN_LIMITATIONS,
        "reproducibility_note": "metric, prediction and report hashes exclude timings; "
        "runtime.json values are wall-clock measurements and vary between runs",
    }
    write_json(paths.manifest, manifest)
    return manifest


KNOWN_LIMITATIONS = [
    (
        "Dataset V1 labels are unreviewed source labels; a disagreement between a model"
        " and a label can be a label error rather than a model error."
    ),
    (
        "Splits are group-disjoint, not contributor-disjoint: one Tatoeba contributor "
        "dominates ckb and the same contributors appear in train and test."
    ),
    (
        "Source and domain are confounded with language (translated prompts only in "
        "sdh; encyclopedic text mostly in ckb/kmr), so internal test scores can reward "
        "source/domain cues rather than language evidence."
    ),
    (
        "Every sdh training record is Arabic script; the strict UD Garrusi benchmark is"
        " Latin script, a combined orthography, variety and domain shift."
    ),
    "The UD Kurmanji slice had prior repository exposure and is exploratory only.",
    (
        "Calibration and thresholds are fitted to the dev distribution; calibrated "
        "scores are estimates, not guaranteed probabilities of correctness, especially "
        "under shift."
    ),
    "Hyperparameter grids are deliberately small; better configurations may exist.",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kurdish-langid-experiment")
    parser.add_argument(
        "phase",
        choices=(
            "preflight",
            "select",
            "evaluate",
            "external",
            "report",
            "reproduce",
            "all",
        ),
    )
    parser.add_argument(
        "--config", default="configs/langid/experiments/baseline-v1.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo selection when a lock exists (refused after test evaluation)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        phases = (
            ("preflight", "select", "evaluate", "external", "report", "reproduce")
            if args.phase == "all"
            else (args.phase,)
        )
        for phase in phases:
            start = time.perf_counter()
            if phase == "preflight":
                run_preflight(config)
            elif phase == "select":
                run_select(config, force=args.force)
            elif phase == "evaluate":
                run_evaluate(config)
            elif phase == "external":
                run_external(config)
            elif phase == "report":
                run_report(config)
            elif phase == "reproduce":
                run_reproduce(config)
            _stderr(f"[{phase}] done in {time.perf_counter() - start:.1f}s")
    except (ProtocolError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

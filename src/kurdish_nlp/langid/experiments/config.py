"""Typed, validated experiment configuration.

All experiment-relevant values (labels, seeds, hyperparameter grids, calibration
and abstention policy, evaluation settings, resource limits and artifact paths)
come from one JSON file. Source modules never hardcode them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.build.config import canonical_json
from kurdish_nlp.langid.schemas import TRAINABLE_LANGUAGE_CODES, LanguageCode

SCHEMA_VERSION = 1
DEV_ROLES = ("dev_tune", "dev_calib")
SPLIT_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration is missing or inconsistent."""


def canonical_sha256(value: Any) -> str:
    """SHA-256 of sorted-key compact UTF-8 JSON (the builder's canonical form)."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HashedPath:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class LengthBucket:
    name: str
    min_tokens: int
    max_tokens: int | None

    def contains(self, tokens: int) -> bool:
        return tokens >= self.min_tokens and (
            self.max_tokens is None or tokens <= self.max_tokens
        )


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    version: str
    build_dir: Path
    build_config: Path
    lock_file: Path
    config_sha256: str
    expected_counts: Mapping[str, int]
    expected_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class DevelopmentProtocol:
    method: str
    group_field: str
    strata: tuple[str, ...]
    roles: Mapping[str, float]
    calibration_crossfit_folds: int


@dataclass(frozen=True, slots=True)
class BenchmarkSlice:
    name: str
    title: str
    expected_language: LanguageCode
    expected_count: int
    allowed_exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSpec:
    benchmark_id: str
    records: HashedPath
    manifest: HashedPath
    strict: BenchmarkSlice
    exploratory: BenchmarkSlice
    restrictions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FastTextSpec:
    artifact: str
    fixed: Mapping[str, Any]
    grid: Mapping[str, tuple[Any, ...]]

    def candidates(self) -> list[dict[str, Any]]:
        """Cartesian grid in deterministic key order, merged over fixed values."""

        keys = sorted(self.grid)
        result = []
        for values in itertools.product(*(self.grid[key] for key in keys)):
            params = dict(self.fixed)
            for key, value in zip(keys, values, strict=True):
                if key == "subwords":
                    params["minn"], params["maxn"] = (int(value[0]), int(value[1]))
                else:
                    params[key] = value
            result.append(params)
        return result


@dataclass(frozen=True, slots=True)
class TfidfSpec:
    artifact: str
    vectorizer_fixed: Mapping[str, Any]
    vectorizer_grid: Mapping[str, tuple[Any, ...]]
    vectorizer_extra_candidates: tuple[Mapping[str, Any], ...]
    classifier_fixed: Mapping[str, Any]
    classifier_grid: Mapping[str, tuple[Any, ...]]

    def vectorizer_candidates(self) -> list[dict[str, Any]]:
        keys = sorted(self.vectorizer_grid)
        seen: list[dict[str, Any]] = []
        grid = (
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.vectorizer_grid[k] for k in keys))
        )
        for item in itertools.chain(grid, self.vectorizer_extra_candidates):
            params = {
                "ngram_range": tuple(int(v) for v in item["ngram_range"]),
                "min_df": item["min_df"],
                "max_features": item["max_features"],
            }
            if params not in seen:
                seen.append(params)
        return seen

    def classifier_candidates(self) -> list[dict[str, Any]]:
        keys = sorted(self.classifier_grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.classifier_grid[k] for k in keys))
        ]


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    method: str
    fit_role: str
    temperature_bounds: tuple[float, float]
    probability_floor: float
    ece_bins: int


@dataclass(frozen=True, slots=True)
class AbstentionSpec:
    fit_role: str
    minimum_letter_count: int
    reference_confidence: float
    reference_margin: float
    confidence_grid: tuple[float, ...]
    margin_grid: tuple[float, ...]
    target_selective_accuracy: float
    minimum_coverage: float


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    focus_pairs: tuple[tuple[LanguageCode, LanguageCode], ...]
    max_per_pair: int
    max_low_confidence_per_label: int
    markdown_examples_per_pair: int


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    primary_metric: str
    length_buckets: tuple[LengthBucket, ...]
    min_slice_support: int
    contributor_top: int
    contributor_head: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    latency_samples: int
    latency_warmup: int
    errors: ErrorSpec

    def length_bucket(self, tokens: int) -> str:
        for bucket in self.length_buckets:
            if bucket.contains(tokens):
                return bucket.name
        return "00"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    max_threads: int
    memory_budget_bytes: int
    max_candidates: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    experiment_id: str
    experiment_version: str
    seed: int
    labels: tuple[LanguageCode, ...]
    focus_labels: tuple[LanguageCode, ...]
    dataset: DatasetSpec
    development: DevelopmentProtocol
    external: ExternalBenchmarkSpec
    fasttext: FastTextSpec
    tfidf: TfidfSpec
    selection_role: str
    selection_metric: str
    selection_tie_breakers: tuple[str, ...]
    calibration: CalibrationSpec
    abstention: AbstentionSpec
    evaluation: EvaluationSpec
    resources: ResourceSpec
    artifact_root: Path

    @property
    def label_values(self) -> tuple[str, ...]:
        return tuple(label.value for label in self.labels)

    def label_index(self) -> dict[str, int]:
        return {label.value: index for index, label in enumerate(self.labels)}

    def model_dir(self, family: str) -> Path:
        return self.artifact_root / family


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ExperimentConfigError(f"{context}: missing required key {key!r}")
    return mapping[key]


def _tuple_grid(grid: Mapping[str, Any], context: str) -> dict[str, tuple[Any, ...]]:
    result = {}
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ExperimentConfigError(f"{context}.{key} must be a non-empty list")
        result[key] = tuple(tuple(v) if isinstance(v, list) else v for v in values)
    return result


def _labels(values: Sequence[str], context: str) -> tuple[LanguageCode, ...]:
    try:
        labels = tuple(LanguageCode(value) for value in values)
    except ValueError as error:
        raise ExperimentConfigError(f"{context}: {error}") from error
    if len(set(labels)) != len(labels):
        raise ExperimentConfigError(f"{context}: duplicate labels")
    if LanguageCode.UND in labels:
        raise ExperimentConfigError(
            f"{context}: und is a runtime abstention result, never a supervised class"
        )
    return labels


def load_experiment_config(
    path: str | Path, *, repo_root: str | Path | None = None
) -> ExperimentConfig:
    """Load and validate an experiment configuration.

    Relative paths inside the configuration resolve against ``repo_root``
    (default: the current working directory).
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ExperimentConfigError(f"experiment configuration not found: {path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(repo_root) if repo_root is not None else Path.cwd()

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"unsupported experiment schema_version: {raw.get('schema_version')!r}"
        )
    labels = _labels(_require(raw, "labels", "config"), "labels")
    if set(labels) != TRAINABLE_LANGUAGE_CODES:
        raise ExperimentConfigError(
            "labels must be exactly the seven trainable language codes"
        )
    focus = _labels(_require(raw, "focus_labels", "config"), "focus_labels")
    if not set(focus) <= set(labels):
        raise ExperimentConfigError("focus_labels must be a subset of labels")
    if raw.get("abstention_label") != LanguageCode.UND.value:
        raise ExperimentConfigError("abstention_label must be 'und'")

    data = _require(raw, "dataset", "config")
    expected_sha = dict(_require(data, "expected_sha256", "dataset"))
    missing = [name for name in SPLIT_FILES if name not in expected_sha]
    if missing:
        raise ExperimentConfigError(f"dataset.expected_sha256 lacks {missing}")
    dataset = DatasetSpec(
        name=data["name"],
        version=data["version"],
        build_dir=resolve(data["build_dir"]),
        build_config=resolve(data["build_config"]),
        lock_file=resolve(data["lock_file"]),
        config_sha256=data["config_sha256"],
        expected_counts=dict(data["expected_counts"]),
        expected_sha256=expected_sha,
    )

    dev = _require(raw, "development_protocol", "config")
    roles = {str(k): float(v) for k, v in dev["roles"].items()}
    if tuple(roles) != DEV_ROLES:
        raise ExperimentConfigError(f"development roles must be {DEV_ROLES}")
    if abs(sum(roles.values()) - 1.0) > 1e-9 or min(roles.values()) <= 0:
        raise ExperimentConfigError(
            "development role ratios must be positive and sum to 1"
        )
    if dev["group_field"] != "document_id":
        raise ExperimentConfigError("development roles must be grouped by document_id")
    folds = int(dev["calibration_crossfit_folds"])
    if folds < 2:
        raise ExperimentConfigError("calibration_crossfit_folds must be at least 2")
    development = DevelopmentProtocol(
        method=dev["method"],
        group_field=dev["group_field"],
        strata=tuple(dev["strata"]),
        roles=roles,
        calibration_crossfit_folds=folds,
    )

    ext = _require(raw, "external_benchmark", "config")

    def benchmark_slice(value: Mapping[str, Any]) -> BenchmarkSlice:
        return BenchmarkSlice(
            name=value["name"],
            title=value["title"],
            expected_language=LanguageCode(value["expected_language"]),
            expected_count=int(value["expected_count"]),
            allowed_exclusion_reasons=tuple(value.get("allowed_exclusion_reasons", ())),
        )

    external = ExternalBenchmarkSpec(
        benchmark_id=ext["benchmark_id"],
        records=HashedPath(resolve(ext["records"]["path"]), ext["records"]["sha256"]),
        manifest=HashedPath(
            resolve(ext["manifest"]["path"]), ext["manifest"]["sha256"]
        ),
        strict=benchmark_slice(ext["strict"]),
        exploratory=benchmark_slice(ext["exploratory"]),
        restrictions=tuple(ext["restrictions"]),
    )
    if "EXPLORATORY" not in external.exploratory.title:
        raise ExperimentConfigError(
            "the exploratory benchmark title must say EXPLORATORY"
        )
    if external.strict.allowed_exclusion_reasons:
        raise ExperimentConfigError("the strict benchmark admits no exclusion reasons")

    models = _require(raw, "models", "config")
    ft = models["fasttext"]
    if ft["fixed"].get("thread") != 1:
        raise ExperimentConfigError(
            "fastText must train with thread=1 for deterministic, reproducible models"
        )
    if ft["fixed"].get("loss") not in {"softmax", "hs", "ova"}:
        raise ExperimentConfigError("unsupported fastText loss")
    fasttext_spec = FastTextSpec(
        artifact=ft["artifact"],
        fixed=dict(ft["fixed"]),
        grid=_tuple_grid(ft["grid"], "models.fasttext.grid"),
    )
    tf = models["tfidf_lr"]
    vfixed = dict(tf["vectorizer_fixed"])
    if vfixed.get("analyzer") not in {"char", "char_wb"}:
        raise ExperimentConfigError("the TF-IDF baseline must use character analyzers")
    if vfixed.get("strip_accents") is not None:
        raise ExperimentConfigError(
            "strip_accents must be null: accent stripping folds Kurdish distinctions"
        )
    tfidf_spec = TfidfSpec(
        artifact=tf["artifact"],
        vectorizer_fixed=vfixed,
        vectorizer_grid=_tuple_grid(
            tf["vectorizer_grid"], "models.tfidf_lr.vectorizer_grid"
        ),
        vectorizer_extra_candidates=tuple(tf.get("vectorizer_extra_candidates", ())),
        classifier_fixed=dict(tf["classifier_fixed"]),
        classifier_grid=_tuple_grid(
            tf["classifier_grid"], "models.tfidf_lr.classifier_grid"
        ),
    )

    sel = _require(raw, "selection", "config")
    if sel["role"] != "dev_tune":
        raise ExperimentConfigError("model selection must use the dev_tune role")

    cal = _require(raw, "calibration", "config")
    if cal["fit_role"] != "dev_calib":
        raise ExperimentConfigError("calibration must be fitted on dev_calib")
    if cal["method"] != "temperature_scaling":
        raise ExperimentConfigError("only temperature_scaling is implemented")
    low, high = (float(v) for v in cal["temperature_bounds"])
    if not 0 < low < 1 < high:
        raise ExperimentConfigError("temperature_bounds must bracket 1")
    calibration = CalibrationSpec(
        method=cal["method"],
        fit_role=cal["fit_role"],
        temperature_bounds=(low, high),
        probability_floor=float(cal["probability_floor"]),
        ece_bins=int(cal["ece_bins"]),
    )

    abst = _require(raw, "abstention", "config")
    if abst["fit_role"] != "dev_calib":
        raise ExperimentConfigError("abstention thresholds must be tuned on dev_calib")
    abstention = AbstentionSpec(
        fit_role=abst["fit_role"],
        minimum_letter_count=int(abst["minimum_letter_count"]),
        reference_confidence=float(abst["reference_policy"]["confidence_threshold"]),
        reference_margin=float(abst["reference_policy"]["margin_threshold"]),
        confidence_grid=tuple(float(v) for v in abst["confidence_grid"]),
        margin_grid=tuple(float(v) for v in abst["margin_grid"]),
        target_selective_accuracy=float(abst["target_selective_accuracy"]),
        minimum_coverage=float(abst["minimum_coverage"]),
    )

    ev = _require(raw, "evaluation", "config")
    if ev["primary_metric"] != "macro_f1":
        raise ExperimentConfigError("the primary metric of this experiment is macro_f1")
    buckets = tuple(
        LengthBucket(b["name"], int(b["min_tokens"]), b["max_tokens"])
        for b in ev["length_buckets"]
    )
    errors = ev["errors"]
    evaluation = EvaluationSpec(
        primary_metric=ev["primary_metric"],
        length_buckets=buckets,
        min_slice_support=int(ev["min_slice_support"]),
        contributor_top=int(ev["contributor_tiers"]["top"]),
        contributor_head=int(ev["contributor_tiers"]["head"]),
        bootstrap_resamples=int(ev["bootstrap"]["resamples"]),
        bootstrap_confidence=float(ev["bootstrap"]["confidence"]),
        latency_samples=int(ev["latency"]["single_text_samples"]),
        latency_warmup=int(ev["latency"]["warmup"]),
        errors=ErrorSpec(
            focus_pairs=tuple(
                (LanguageCode(a), LanguageCode(b)) for a, b in errors["focus_pairs"]
            ),
            max_per_pair=int(errors["max_per_pair"]),
            max_low_confidence_per_label=int(errors["max_low_confidence_per_label"]),
            markdown_examples_per_pair=int(errors["markdown_examples_per_pair"]),
        ),
    )

    res = _require(raw, "resources", "config")
    resources = ResourceSpec(
        max_threads=int(res["max_threads"]),
        memory_budget_bytes=int(res["memory_budget_bytes"]),
        max_candidates=dict(res["max_candidates"]),
    )
    config = ExperimentConfig(
        path=config_path,
        sha256=canonical_sha256(raw),
        raw=raw,
        experiment_id=raw["experiment_id"],
        experiment_version=raw["experiment_version"],
        seed=int(raw["seed"]),
        labels=labels,
        focus_labels=focus,
        dataset=dataset,
        development=development,
        external=external,
        fasttext=fasttext_spec,
        tfidf=tfidf_spec,
        selection_role=sel["role"],
        selection_metric=sel["metric"],
        selection_tie_breakers=tuple(sel["tie_breakers"]),
        calibration=calibration,
        abstention=abstention,
        evaluation=evaluation,
        resources=resources,
        artifact_root=resolve(raw["artifacts"]["root"]),
    )
    _check_search_bounds(config)
    return config


def _check_search_bounds(config: ExperimentConfig) -> None:
    ft = len(config.fasttext.candidates())
    tf = len(config.tfidf.vectorizer_candidates()) * len(
        config.tfidf.classifier_candidates()
    )
    for family, size in (("fasttext", ft), ("tfidf_lr", tf)):
        bound = config.resources.max_candidates.get(family)
        if bound is None or size > bound:
            raise ExperimentConfigError(
                f"{family} search has {size} candidates, above the configured "
                f"bound {bound}; reduce the grid"
            )

"""Language ID service over the frozen experiment artifacts.

Models are loaded once through ``kurdish_nlp.langid.create_detector`` (which
verifies each artifact against its inference spec; the TF-IDF pickle is also
checked against its sidecar hash and scikit-learn version before unpickling).
Only the allowlisted families under the configured artifacts directory are ever
loaded, and each inference spec must match the hash recorded in the selection
lock when the lock is present. Calibration and abstention are entirely the
package's; this module only formats its ``DetectionResult``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kurdish_nlp.langid import AbstentionReason, DetectionResult, create_detector
from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.build.textkeys import script_class
from kurdish_nlp.langid.experiments.detector import (
    INFERENCE_FILE,
    CalibratedLanguageDetector,
)
from kurdish_nlp.langid.normalization import lexical_letter_count
from kurdish_nlp.langid.reliability import ReliabilityPolicy

from app.errors import ApiError
from app.languages import SUPPORTED_CODES, language_info
from app.schemas.common import Language
from app.schemas.detect import (
    ClassScore,
    DetectResponse,
    InputStats,
    ModelRef,
    Notice,
    PolicyInfo,
)
from app.schemas.models import ExternalBenchmark, InternalTestMetrics, ModelInfo

logger = logging.getLogger("kurdish_nlp.api.langid")

LOCK_FILE = "selection.lock.json"
MANIFEST_FILE = "experiment_manifest.json"

# The only model families the server will load, in display order.
FAMILIES: dict[str, dict[str, str]] = {
    "tfidf_lr": {
        "name": "TF-IDF + LR",
        "full_name": "TF-IDF + Logistic Regression",
        "framework": "scikit-learn",
    },
    "fasttext": {
        "name": "fastText",
        "full_name": "fastText",
        "framework": "fastText",
    },
}

REASON_MESSAGES: dict[str, str] = {
    AbstentionReason.EMPTY_TEXT: "There is no text to identify.",
    AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT: (
        "Too few letters to identify a language. Add more text."
    ),
    AbstentionReason.LOW_CONFIDENCE: "The top score is below the policy's confidence threshold.",
    AbstentionReason.LOW_MARGIN: "The top two languages scored too close to call.",
    AbstentionReason.NO_SUPPORTED_PREDICTIONS: "The model returned no supported language.",
}

# Grounded in reports/comparison.md for baseline-v1: most internal-test errors
# fall in the 1-5 token bucket, and the strict UD Garrusi benchmark (Latin-script
# Southern Kurdish, absent from training) is mostly predicted as kmr.
SHORT_TEXT_WORDS = 5
SHORT_TEXT_NOTE = "Short input. Most internal-test errors were on texts of five words or fewer."
LATIN_SDH_NOTE = (
    "Southern Kurdish in Latin script is not in the training data; "
    "the models usually label it Kurmanji."
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("could not read %s", path.name)
        return None


def _public_reason(error: Exception) -> str:
    """A short, path-free explanation of why a model did not load."""

    if isinstance(error, FileNotFoundError):
        return "Model files are missing."
    if isinstance(error, ImportError):
        return "A required library is not installed."
    if isinstance(error, RuntimeError):
        text = str(error)
        if "scikit-learn" in text:
            return text.split(";")[0].capitalize() + "."
        if "fastText" in text or "fasttext" in text:
            return "fastText is not installed."
        return "The model could not be loaded."
    if isinstance(error, ValueError) and "sha256" in str(error):
        return "Model file failed its integrity check."
    return "The model could not be loaded."


@dataclass
class LoadedModel:
    family: str
    directory: Path
    spec: dict[str, Any] | None = None
    lock_entry: dict[str, Any] | None = None
    tuned: CalibratedLanguageDetector | None = None
    reference: CalibratedLanguageDetector | None = None
    reasons: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.tuned is not None


class LanguageIdService:
    def __init__(self, artifacts_dir: Path, default_model: str) -> None:
        self.artifacts_dir = artifacts_dir
        self.preferred_default = default_model
        self.experiment_lock = _read_json(artifacts_dir / LOCK_FILE)
        dataset = (_read_json(artifacts_dir / MANIFEST_FILE) or {}).get("dataset") or {}
        self.dataset_label = (
            " ".join(str(part) for part in (dataset.get("name"), dataset.get("version")) if part)
            or "unknown dataset"
        )
        self.models: dict[str, LoadedModel] = {
            family: LoadedModel(family, artifacts_dir / family) for family in FAMILIES
        }
        self._loaded = False
        self._load_guard = threading.Lock()
        self._descriptions: list[ModelInfo] | None = None

    # Loading -----------------------------------------------------------------

    def load(self) -> None:
        with self._load_guard:
            if self._loaded:
                return
            for model in self.models.values():
                self._load_one(model)
            self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _load_one(self, model: LoadedModel) -> None:
        spec_path = model.directory / INFERENCE_FILE
        model.spec = _read_json(spec_path)
        if model.spec is None:
            model.reasons.append("Model files are missing.")
            return
        if self.experiment_lock is not None:
            model.lock_entry = self.experiment_lock.get("families", {}).get(model.family)
            expected = (model.lock_entry or {}).get("inference_spec_sha256")
            if expected is None or file_sha256(spec_path) != expected:
                model.reasons.append("Inference spec does not match the selection lock.")
                return
        try:
            started = time.perf_counter()
            tuned = create_detector(backend="experiment", model_path=model.directory)
        except Exception as error:  # noqa: BLE001 - reported as unavailable
            logger.warning("model %s unavailable: %s", model.family, error)
            model.reasons.append(_public_reason(error))
            return
        assert isinstance(tuned, CalibratedLanguageDetector)
        model.tuned = tuned
        model.reference = self._reference_detector(tuned, model.spec)
        logger.info("loaded %s in %.2fs", tuned.model_identifier, time.perf_counter() - started)

    @staticmethod
    def _reference_detector(
        tuned: CalibratedLanguageDetector, spec: dict[str, Any]
    ) -> CalibratedLanguageDetector:
        """Same as ``load_experiment_detector(calibrated=False)``, sharing the classifier."""

        policy = spec["reference_policy"]
        return CalibratedLanguageDetector(
            tuned.classifier,
            temperature=1.0,
            policy=ReliabilityPolicy(
                confidence_threshold=policy["confidence_threshold"],
                margin_threshold=policy["margin_threshold"],
                minimum_letter_count=policy["minimum_letter_count"],
            ),
            probability_floor=spec["probability_floor"],
            model_identifier=spec["model_identifier"],
            model_version=spec["model_version"],
        )

    # Discovery ---------------------------------------------------------------

    @property
    def default_model(self) -> str | None:
        preferred = self.models.get(self.preferred_default)
        if preferred is not None and preferred.ready:
            return preferred.family
        return next((m.family for m in self.models.values() if m.ready), None)

    @property
    def any_ready(self) -> bool:
        return any(model.ready for model in self.models.values())

    def describe(self) -> list[ModelInfo]:
        """Model metadata; artifacts are frozen, so it is computed once after loading."""

        if self._descriptions is None or not self._loaded:
            default = self.default_model
            self._descriptions = [
                self._describe(model, model.family == default) for model in self.models.values()
            ]
        return [info.model_copy(deep=True) for info in self._descriptions]

    def _describe(self, model: LoadedModel, is_default: bool) -> ModelInfo:
        meta = FAMILIES[model.family]
        spec = model.spec or {}
        lock = model.lock_entry or {}
        info = ModelInfo(
            id=model.family,
            name=meta["name"],
            full_name=meta["full_name"],
            task="language",
            description=self._description(model.family, lock.get("selected_params")),
            languages=list(spec.get("labels") or SUPPORTED_CODES),
            status="ready" if model.ready else "unavailable",
            default=is_default,
            unavailable_reasons=list(model.reasons),
            identifier=spec.get("model_identifier"),
            version=spec.get("model_version"),
            experiment=(self.experiment_lock or {}).get("experiment_id"),
            framework=meta["framework"],
            size_bytes=lock.get("artifact_bytes"),
            temperature=spec.get("temperature"),
            policies=self._policies(spec),
            artifact_sha256=spec.get("artifact_sha256"),
        )
        if model.ready:
            info.internal_test = self._internal_test(model)
            info.external_benchmarks = self._external(model)
        return info

    @staticmethod
    def _policies(spec: dict[str, Any]) -> list[PolicyInfo]:
        policies = []
        for name, key, calibrated in (
            ("tuned", "policy", True),
            ("reference", "reference_policy", False),
        ):
            if key in spec:
                policies.append(PolicyInfo(name=name, calibrated=calibrated, **spec[key]))
        return policies

    @staticmethod
    def _description(family: str, params: dict[str, Any] | None) -> str:
        if family == "tfidf_lr":
            low, high = ((params or {}).get("vectorizer") or {}).get("ngram_range") or (None, None)
            grams = f"{low}–{high}-gram " if low and high else ""
            return (
                f"Character {grams}TF-IDF features with multinomial logistic regression. "
                "Temperature-calibrated."
            )
        low, high = (params or {}).get("minn"), (params or {}).get("maxn")
        grams = f" with character {low}–{high}-grams" if low and high else ""
        return f"fastText supervised classifier{grams}. Temperature-calibrated."

    def _internal_test(self, model: LoadedModel) -> InternalTestMetrics | None:
        evaluation = _read_json(model.directory / "evaluation.json") or {}
        test = ((evaluation.get("roles") or {}).get("test") or {}).get("classification")
        if not test or "macro_f1" not in test:
            return None
        return InternalTestMetrics(
            dataset=self.dataset_label,
            records=test.get("records"),
            macro_f1=test["macro_f1"],
            accuracy=test.get("accuracy"),
        )

    @staticmethod
    def _external(model: LoadedModel) -> list[ExternalBenchmark]:
        external = _read_json(model.directory / "evaluation_external.json") or {}
        strict = external.get("strict")
        if not strict or "argmax" not in strict:
            return []
        return [
            ExternalBenchmark(
                name="UD v2.18 Southern Kurdish (Garrusi)",
                description=(
                    "Written in Latin script; all Southern Kurdish training data is in Arabic script."
                ),
                records=strict.get("records"),
                expected_label=strict.get("expected_language", "sdh"),
                recall=strict["argmax"]["recall"],
                correct=strict["argmax"].get("correct"),
            )
        ]

    # Inference ---------------------------------------------------------------

    def check_known(self, model_id: str | None) -> None:
        if model_id is not None and model_id not in self.models:
            raise ApiError(
                422,
                "unknown_model",
                f"Unknown model '{model_id[:40]}'.",
                {"available": list(self.models)},
            )

    def resolve(self, model_id: str | None) -> LoadedModel:
        if model_id is None:
            default = self.default_model
            if default is None:
                raise ApiError(
                    503,
                    "model_unavailable",
                    "No language ID model is available.",
                )
            return self.models[default]
        self.check_known(model_id)
        model = self.models[model_id]
        if not model.ready:
            raise ApiError(
                503,
                "model_unavailable",
                f"{FAMILIES[model.family]['name']} is unavailable.",
                {"model": model.family, "reasons": model.reasons},
            )
        return model

    def detect(self, text: str, model: LoadedModel, policy: str, top_k: int) -> DetectResponse:
        detector = model.tuned if policy == "tuned" else model.reference
        assert detector is not None
        started = time.perf_counter()
        with model.lock:
            result = detector.detect(text, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000
        return self._response(text, result, model, detector, policy, elapsed)

    def _response(
        self,
        text: str,
        result: DetectionResult,
        model: LoadedModel,
        detector: CalibratedLanguageDetector,
        policy: str,
        elapsed_ms: float,
    ) -> DetectResponse:
        info = language_info(result.language.value)
        reason = result.reason.value if result.reason else None
        stats = InputStats(
            characters=len(text),
            letters=lexical_letter_count(text),
            words=len(text.split()),
            script=script_class(text),
        )
        top = result.alternatives[0].language.value if result.alternatives else None
        warnings: list[Notice] = []
        if top == "kmr" and stats.script == "latin":
            warnings.append(
                Notice(code="latin_script_sdh", severity="info", message=LATIN_SDH_NOTE)
            )
        if result.alternatives and stats.words <= SHORT_TEXT_WORDS:
            warnings.append(Notice(code="short_text", severity="info", message=SHORT_TEXT_NOTE))

        spec = model.spec or {}
        return DetectResponse(
            language=Language(code=info.code, name=info.name, alt_name=info.alt_name),
            is_reliable=result.is_reliable,
            abstained=not result.is_reliable,
            reason=reason,
            reason_message=REASON_MESSAGES.get(reason) if reason else None,
            confidence=result.confidence,
            margin=result.margin,
            scores=[
                ClassScore(
                    code=item.language.value,
                    name=language_info(item.language.value).name,
                    score=item.confidence,
                )
                for item in result.alternatives
            ],
            model=ModelRef(
                id=model.family,
                name=FAMILIES[model.family]["name"],
                identifier=result.model_identifier,
                version=result.model_version,
            ),
            policy=PolicyInfo(
                name=policy,  # type: ignore[arg-type]
                calibrated=policy == "tuned",
                confidence_threshold=detector.policy.confidence_threshold,
                margin_threshold=detector.policy.margin_threshold,
                minimum_letter_count=detector.policy.minimum_letter_count,
            ),
            score_semantics=spec.get(
                "score_semantics", "model score estimates; not guaranteed probabilities"
            )
            if policy == "tuned"
            else "raw model scores; not calibrated and not guaranteed probabilities",
            warnings=warnings,
            input=stats,
            elapsed_ms=round(elapsed_ms, 3),
        )

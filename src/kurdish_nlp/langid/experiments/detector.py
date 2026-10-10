"""Runtime detector for experiment artifacts (fastText or TF-IDF + LR).

``load_experiment_detector(model_dir)`` reads the frozen ``inference.json``
written at selection time (family, artifact hash, label order, temperature and
abstention thresholds), verifies the artifact hash and returns a
:class:`CalibratedLanguageDetector`. Detection follows the existing contract:
conservative normalization, then scores, then ``ReliabilityPolicy``, then a
``DetectionResult`` whose ``und`` means abstention, never a trained class.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kurdish_nlp.langid.base import LanguageDetector
from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.experiments.calibration import apply_temperature
from kurdish_nlp.langid.experiments.evaluation import ranked
from kurdish_nlp.langid.normalization import normalize_text
from kurdish_nlp.langid.reliability import ReliabilityDecision, ReliabilityPolicy
from kurdish_nlp.langid.schemas import (
    TRAINABLE_LANGUAGE_CODES,
    AbstentionReason,
    DetectionResult,
    LanguageCode,
    Prediction,
)

INFERENCE_FILE = "inference.json"
_PRE_MODEL_REASONS = frozenset(
    {AbstentionReason.EMPTY_TEXT, AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT}
)


class ProbabilisticClassifier(Protocol):
    labels: tuple[str, ...]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


class CalibratedLanguageDetector(LanguageDetector):
    """Calibrated scores plus the project's abstention policy."""

    def __init__(
        self,
        classifier: ProbabilisticClassifier,
        *,
        temperature: float,
        policy: ReliabilityPolicy,
        probability_floor: float,
        model_identifier: str,
        model_version: str,
    ) -> None:
        labels = tuple(classifier.labels)
        if {LanguageCode(label) for label in labels} != TRAINABLE_LANGUAGE_CODES:
            raise ValueError("a detector needs exactly the seven trainable labels")
        self.classifier = classifier
        self.labels = labels
        self.temperature = float(temperature)
        self.policy = policy
        self.probability_floor = float(probability_floor)
        self.model_identifier = model_identifier
        self.model_version = model_version

    def predict_proba_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Calibrated score matrix (columns in ``self.labels`` order)."""

        normalized = [normalize_text(text) for text in texts]
        if any(not text for text in normalized):
            raise ValueError("cannot score empty text; detect() abstains on it")
        raw = self.classifier.predict_proba(normalized)
        if self.temperature == 1.0:
            return raw
        return apply_temperature(raw, self.temperature, self.probability_floor)

    def predict_proba(self, text: str) -> dict[str, float]:
        """Calibrated score per label for one text, highest first."""

        row = self.predict_proba_batch([text])[0]
        return {self.labels[j]: float(row[j]) for j in ranked(row[None, :])[0]}

    def detect(self, text: str, top_k: int = 3) -> DetectionResult:
        return self.detect_batch([text], top_k=top_k)[0]

    def detect_batch(
        self, texts: Sequence[str], top_k: int = 3
    ) -> list[DetectionResult]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        normalized = [normalize_text(text) for text in texts]
        results: list[DetectionResult | None] = [None] * len(normalized)
        to_score = []
        for i, text in enumerate(normalized):
            preliminary = self.policy.evaluate(text, ())
            if preliminary.reason in _PRE_MODEL_REASONS:
                results[i] = self._result((), preliminary, top_k)
            else:
                to_score.append(i)
        if to_score:
            raw = self.classifier.predict_proba([normalized[i] for i in to_score])
            scores = (
                raw
                if self.temperature == 1.0
                else apply_temperature(raw, self.temperature, self.probability_floor)
            )
            order = ranked(scores)
            for row, i in enumerate(to_score):
                predictions = tuple(
                    Prediction(
                        LanguageCode(self.labels[j]),
                        min(1.0, max(0.0, float(scores[row, j]))),
                    )
                    for j in order[row]
                )
                decision = self.policy.evaluate(normalized[i], predictions)
                results[i] = self._result(predictions, decision, top_k)
        return [result for result in results if result is not None]

    def _result(
        self,
        predictions: tuple[Prediction, ...],
        decision: ReliabilityDecision,
        top_k: int,
    ) -> DetectionResult:
        return DetectionResult(
            language=decision.language,
            confidence=predictions[0].confidence if predictions else 0.0,
            is_reliable=decision.is_reliable,
            alternatives=predictions[:top_k],
            margin=decision.margin,
            reason=decision.reason,
            model_identifier=self.model_identifier,
            model_version=self.model_version,
        )


def write_inference_spec(model_dir: Path, spec: dict[str, Any]) -> Path:
    path = model_dir / INFERENCE_FILE
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_classifier(family: str, artifact: Path, labels: Sequence[str]):
    if family == "fasttext":
        from kurdish_nlp.langid.experiments.fasttext_baseline import FastTextModel

        return FastTextModel.load(artifact, labels)
    if family == "tfidf_lr":
        from kurdish_nlp.langid.experiments.tfidf_baseline import TfidfLogisticModel

        return TfidfLogisticModel.load(artifact, labels)
    raise ValueError(f"unknown experiment model family: {family!r}")


def load_experiment_detector(
    model_dir: str | Path, *, calibrated: bool = True
) -> CalibratedLanguageDetector:
    """Load a frozen experiment model directory as a ``LanguageDetector``.

    ``calibrated=False`` uses raw scores with the project's reference policy
    thresholds recorded in the same file (for comparison only).
    """

    directory = Path(model_dir)
    spec_path = directory / INFERENCE_FILE
    if not spec_path.is_file():
        raise FileNotFoundError(f"experiment inference spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    artifact = directory / spec["artifact"]
    if not artifact.is_file():
        raise FileNotFoundError(f"experiment model artifact not found: {artifact}")
    actual = file_sha256(artifact)
    if actual != spec["artifact_sha256"]:
        raise ValueError(
            f"{artifact} sha256 {actual} differs from the frozen {spec['artifact_sha256']}"
        )
    classifier = load_classifier(spec["family"], artifact, spec["labels"])
    policy_spec = spec["policy"] if calibrated else spec["reference_policy"]
    return CalibratedLanguageDetector(
        classifier,
        temperature=spec["temperature"] if calibrated else 1.0,
        policy=ReliabilityPolicy(
            confidence_threshold=policy_spec["confidence_threshold"],
            margin_threshold=policy_spec["margin_threshold"],
            minimum_letter_count=policy_spec["minimum_letter_count"],
        ),
        probability_floor=spec["probability_floor"],
        model_identifier=spec["model_identifier"],
        model_version=spec["model_version"],
    )

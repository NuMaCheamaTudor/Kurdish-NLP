"""Optional fastText runtime adapter.

Importing this module does not import fastText or access the network. The optional
dependency is imported only when a detector instance loads a real model file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from kurdish_nlp.langid.base import LanguageDetector
from kurdish_nlp.langid.normalization import normalize_text
from kurdish_nlp.langid.reliability import ReliabilityDecision, ReliabilityPolicy
from kurdish_nlp.langid.schemas import (
    TRAINABLE_LANGUAGE_CODES,
    AbstentionReason,
    DetectionResult,
    LanguageCode,
    Prediction,
)


class FastTextModel(Protocol):
    def predict(
        self,
        text: str,
        k: int = 1,
    ) -> tuple[Sequence[str], Sequence[float]]: ...


ModelLoader = Callable[[Path], FastTextModel]

# fastText reports each probability as exp(log(p + 1e-5)) (its std_log adds
# 1e-5), so raw scores exceed the softmax value by 1e-5 and can exceed 1.0.
FASTTEXT_PROBABILITY_OFFSET = 1e-5


def fasttext_probability(raw_score: float) -> float:
    """Undo fastText's log offset and clip the score into [0, 1]."""

    return min(1.0, max(0.0, float(raw_score) - FASTTEXT_PROBABILITY_OFFSET))


class NumpyCompatibleFastTextModel:
    """Route single-text prediction through fastText's list API.

    fastText 0.9.3's single-string ``predict`` calls ``np.array(..., copy=False)``,
    which NumPy 2 rejects; the list path returns arrays directly. Scores are
    returned without fastText's 1e-5 offset so they are valid confidences.
    """

    def __init__(self, model: object) -> None:
        self._model = model

    def predict(self, text: str, k: int = 1) -> tuple[Sequence[str], Sequence[float]]:
        labels, scores = self._model.predict([text], k=k)  # type: ignore[attr-defined]
        return tuple(labels[0]), tuple(fasttext_probability(s) for s in scores[0])


class FastTextLanguageDetector(LanguageDetector):
    """Adapt a custom fastText classifier to the common detector interface."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.70,
        margin_threshold: float = 0.20,
        minimum_letter_count: int = 3,
        model_identifier: str = "kurdish-langid-fasttext",
        model_version: str = "unversioned",
        model_loader: ModelLoader | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"fastText model not found: {path}")

        self.model_path = path
        self.model_identifier = model_identifier
        self.model_version = model_version
        self.policy = ReliabilityPolicy(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            minimum_letter_count=minimum_letter_count,
        )
        loader = model_loader or self._load_fasttext_model
        self._model = loader(path)

    @staticmethod
    def _load_fasttext_model(path: Path) -> FastTextModel:
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "fastText support is optional; install the project with "
                "the 'fasttext' extra to load a fastText model"
            ) from error
        return NumpyCompatibleFastTextModel(fasttext.load_model(str(path)))

    def detect(self, text: str, top_k: int = 3) -> DetectionResult:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        normalized = normalize_text(text)
        preliminary = self.policy.evaluate(normalized, ())
        if preliminary.reason in {
            AbstentionReason.EMPTY_TEXT,
            AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT,
        }:
            return self._result((), preliminary)

        requested_k = min(
            max(top_k, 2),
            len(TRAINABLE_LANGUAGE_CODES),
        )
        labels, scores = self._model.predict(normalized, k=requested_k)
        predictions = self._parse_predictions(labels, scores)
        decision = self.policy.evaluate(normalized, predictions)
        return self._result(predictions[:top_k], decision, predictions)

    @staticmethod
    def _parse_predictions(
        labels: Sequence[str],
        scores: Sequence[float],
    ) -> tuple[Prediction, ...]:
        candidates: dict[LanguageCode, Prediction] = {}
        for raw_label, raw_score in zip(labels, scores):
            label = (
                raw_label.decode("utf-8")
                if isinstance(raw_label, bytes)
                else str(raw_label)
            )
            label = label.removeprefix("__label__")
            try:
                language = LanguageCode(label)
            except ValueError:
                continue
            if language not in TRAINABLE_LANGUAGE_CODES:
                continue
            prediction = Prediction(language=language, confidence=float(raw_score))
            previous = candidates.get(language)
            if previous is None or prediction.confidence > previous.confidence:
                candidates[language] = prediction

        return tuple(
            sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)
        )

    def _result(
        self,
        visible_predictions: tuple[Prediction, ...],
        decision: ReliabilityDecision,
        scored_predictions: tuple[Prediction, ...] | None = None,
    ) -> DetectionResult:
        # ``scored_predictions`` retains the second candidate used for the margin
        # when a caller asks to display only top_k=1.
        ranked = (
            scored_predictions
            if scored_predictions is not None
            else visible_predictions
        )
        confidence = ranked[0].confidence if ranked else 0.0
        return DetectionResult(
            language=decision.language,
            confidence=confidence,
            is_reliable=decision.is_reliable,
            alternatives=visible_predictions,
            margin=decision.margin,
            reason=decision.reason,
            model_identifier=self.model_identifier,
            model_version=self.model_version,
        )

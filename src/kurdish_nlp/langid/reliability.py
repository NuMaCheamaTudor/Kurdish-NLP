"""Isolated abstention policy for language detector backends."""

from __future__ import annotations

from dataclasses import dataclass

from kurdish_nlp.langid.normalization import lexical_letter_count
from kurdish_nlp.langid.schemas import AbstentionReason, LanguageCode, Prediction


@dataclass(frozen=True, slots=True)
class ReliabilityDecision:
    language: LanguageCode
    is_reliable: bool
    reason: AbstentionReason | None
    margin: float


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    """Threshold policy applied to ranked, uncalibrated backend scores."""

    confidence_threshold: float = 0.70
    margin_threshold: float = 0.20
    minimum_letter_count: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("margin_threshold", self.margin_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.minimum_letter_count < 1:
            raise ValueError("minimum_letter_count must be at least 1")

    def evaluate(
        self,
        normalized_text: str,
        predictions: tuple[Prediction, ...],
    ) -> ReliabilityDecision:
        if not normalized_text:
            return self._abstain(AbstentionReason.EMPTY_TEXT)
        if lexical_letter_count(normalized_text) < self.minimum_letter_count:
            return self._abstain(AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT)
        if not predictions:
            return self._abstain(AbstentionReason.NO_SUPPORTED_PREDICTIONS)

        top_score = predictions[0].confidence
        second_score = predictions[1].confidence if len(predictions) > 1 else 0.0
        margin = max(0.0, min(1.0, top_score - second_score))

        if top_score < self.confidence_threshold:
            return self._abstain(AbstentionReason.LOW_CONFIDENCE, margin)
        if margin < self.margin_threshold:
            return self._abstain(AbstentionReason.LOW_MARGIN, margin)
        return ReliabilityDecision(
            language=predictions[0].language,
            is_reliable=True,
            reason=None,
            margin=margin,
        )

    @staticmethod
    def _abstain(
        reason: AbstentionReason,
        margin: float = 0.0,
    ) -> ReliabilityDecision:
        return ReliabilityDecision(
            language=LanguageCode.UND,
            is_reliable=False,
            reason=reason,
            margin=margin,
        )

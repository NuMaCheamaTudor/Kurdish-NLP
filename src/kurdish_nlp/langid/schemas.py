"""Typed runtime schemas for language identification.

The ``confidence`` values are backend scores. In the initial fastText adapter they
are raw fastText scores, not calibrated probabilities. A future calibration layer
can preserve these public schemas while changing how scores are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from math import isfinite
from typing import Any


class LanguageCode(StrEnum):
    """Canonical runtime language and dialect codes."""

    CKB = "ckb"
    KMR = "kmr"
    SDH = "sdh"
    AR = "ar"
    FA = "fa"
    TR = "tr"
    EN = "en"
    UND = "und"


TRAINABLE_LANGUAGE_CODES = frozenset(
    code for code in LanguageCode if code is not LanguageCode.UND
)


class AbstentionReason(StrEnum):
    """Why a detector returned ``und``."""

    EMPTY_TEXT = "empty_text"
    INSUFFICIENT_LEXICAL_CONTENT = "insufficient_lexical_content"
    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    NO_SUPPORTED_PREDICTIONS = "no_supported_predictions"


def _validate_score(value: float, field_name: str) -> float:
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be a finite value between 0 and 1")
    return score


@dataclass(frozen=True, slots=True)
class Prediction:
    """One ranked candidate and its backend confidence score."""

    language: LanguageCode
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.language, LanguageCode):
            object.__setattr__(self, "language", LanguageCode(self.language))
        object.__setattr__(
            self,
            "confidence",
            _validate_score(self.confidence, "confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language.value, "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Backend-independent language identification result.

    For an abstention, ``language`` is ``und`` while ``confidence`` remains the
    score of the strongest candidate, if one exists. ``reason`` explains why that
    candidate was not considered reliable.
    """

    language: LanguageCode
    confidence: float
    is_reliable: bool
    alternatives: tuple[Prediction, ...]
    margin: float
    reason: AbstentionReason | None
    model_identifier: str
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.language, LanguageCode):
            object.__setattr__(self, "language", LanguageCode(self.language))
        if self.reason is not None and not isinstance(self.reason, AbstentionReason):
            object.__setattr__(self, "reason", AbstentionReason(self.reason))

        object.__setattr__(
            self,
            "confidence",
            _validate_score(self.confidence, "confidence"),
        )
        object.__setattr__(self, "margin", _validate_score(self.margin, "margin"))

        alternatives = tuple(self.alternatives)
        object.__setattr__(self, "alternatives", alternatives)
        if any(
            left.confidence < right.confidence for left, right in pairwise(alternatives)
        ):
            raise ValueError("alternatives must be ranked by descending confidence")
        if len({item.language for item in alternatives}) != len(alternatives):
            raise ValueError("alternatives must not contain duplicate languages")
        if alternatives and self.confidence != alternatives[0].confidence:
            raise ValueError("confidence must match the highest-ranked alternative")
        if not alternatives and self.confidence != 0.0:
            raise ValueError("confidence must be 0 when alternatives are empty")

        if not self.model_identifier.strip() or not self.model_version.strip():
            raise ValueError("model identifier and version must be non-empty")
        if self.is_reliable:
            if self.language is LanguageCode.UND or self.reason is not None:
                raise ValueError("a reliable result cannot be und or have a reason")
            if not alternatives or alternatives[0].language is not self.language:
                raise ValueError("a reliable language must match the top alternative")
        elif self.language is not LanguageCode.UND or self.reason is None:
            raise ValueError("an unreliable result must be und and include a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language.value,
            "confidence": self.confidence,
            "is_reliable": self.is_reliable,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "margin": self.margin,
            "reason": self.reason.value if self.reason else None,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
        }

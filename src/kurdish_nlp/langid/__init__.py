"""Stable language-identification interfaces and implementations."""

from kurdish_nlp.langid.base import LanguageDetector
from kurdish_nlp.langid.registry import create_detector
from kurdish_nlp.langid.schemas import (
    AbstentionReason,
    DetectionResult,
    LanguageCode,
    Prediction,
)

__all__ = [
    "AbstentionReason",
    "DetectionResult",
    "LanguageCode",
    "LanguageDetector",
    "Prediction",
    "create_detector",
]

"""Public package for the Kurdish NLP project."""

from kurdish_nlp.langid import (
    DetectionResult,
    LanguageCode,
    LanguageDetector,
    Prediction,
    create_detector,
)

__all__ = [
    "DetectionResult",
    "LanguageCode",
    "LanguageDetector",
    "Prediction",
    "create_detector",
]

__version__ = "0.1.0"

"""Small detector factory for service-layer construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kurdish_nlp.langid.base import LanguageDetector


def create_detector(
    *,
    backend: str,
    model_path: str | Path,
    **options: Any,
) -> LanguageDetector:
    """Create a configured detector without exposing backend details to callers."""

    normalized_backend = backend.strip().lower()
    if normalized_backend == "fasttext":
        from kurdish_nlp.langid.fasttext_detector import FastTextLanguageDetector

        return FastTextLanguageDetector(model_path=model_path, **options)
    raise ValueError(f"unsupported language detector backend: {backend!r}")

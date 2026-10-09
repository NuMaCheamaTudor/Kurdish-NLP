"""Backend-independent language detector interface."""

from abc import ABC, abstractmethod

from kurdish_nlp.langid.schemas import DetectionResult


class LanguageDetector(ABC):
    """Stable interface consumed by future service and API layers."""

    @abstractmethod
    def detect(self, text: str, top_k: int = 3) -> DetectionResult:
        """Detect the language or return ``und`` when evidence is insufficient."""

from pathlib import Path

import pytest

from kurdish_nlp.langid.fasttext_detector import FastTextLanguageDetector
from kurdish_nlp.langid.registry import create_detector


class FakeModel:
    def predict(self, text: str, k: int = 1):
        return ("__label__en",), (0.99,)


def test_registry_creates_fasttext_detector(tmp_path: Path) -> None:
    model_path = tmp_path / "fixture.bin"
    model_path.write_bytes(b"fixture")
    detector = create_detector(
        backend="FASTTEXT",
        model_path=model_path,
        model_loader=lambda _: FakeModel(),
    )
    assert isinstance(detector, FastTextLanguageDetector)


def test_registry_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        create_detector(backend="transformer", model_path=tmp_path / "model")

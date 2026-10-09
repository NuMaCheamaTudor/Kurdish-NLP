from pathlib import Path

import pytest

from kurdish_nlp.langid.fasttext_detector import FastTextLanguageDetector
from kurdish_nlp.langid.schemas import AbstentionReason, LanguageCode


class FakeFastTextModel:
    def __init__(self, labels: tuple[str, ...], scores: tuple[float, ...]) -> None:
        self.labels = labels
        self.scores = scores
        self.calls: list[tuple[str, int]] = []

    def predict(self, text: str, k: int = 1):
        self.calls.append((text, k))
        return self.labels[:k], self.scores[:k]


def _detector(tmp_path: Path, model: FakeFastTextModel, **options):
    model_path = tmp_path / "fixture.bin"
    model_path.write_bytes(b"test fixture, not a model")
    loads = []

    def loader(path: Path):
        loads.append(path)
        return model

    detector = FastTextLanguageDetector(
        model_path,
        model_loader=loader,
        model_version="test",
        **options,
    )
    assert loads == [model_path]
    return detector


def test_reliable_detection_and_top_k(tmp_path: Path) -> None:
    model = FakeFastTextModel(
        ("__label__ckb", "__label__fa", "__label__ar"),
        (0.94, 0.04, 0.02),
    )
    result = _detector(tmp_path, model).detect("  دەقێکی\nکوردی  ", top_k=3)
    assert result.language is LanguageCode.CKB
    assert result.margin == pytest.approx(0.90)
    assert [item.language for item in result.alternatives] == [
        LanguageCode.CKB,
        LanguageCode.FA,
        LanguageCode.AR,
    ]
    assert model.calls == [("دەقێکی کوردی", 3)]


def test_top_k_one_still_requests_two_scores_for_margin(tmp_path: Path) -> None:
    model = FakeFastTextModel(("__label__en", "__label__tr"), (0.90, 0.30))
    result = _detector(tmp_path, model).detect("valid text", top_k=1)
    assert len(result.alternatives) == 1
    assert result.margin == pytest.approx(0.60)
    assert model.calls[0][1] == 2


def test_empty_and_short_text_do_not_call_model(tmp_path: Path) -> None:
    model = FakeFastTextModel(("__label__en",), (0.99,))
    detector = _detector(tmp_path, model)
    empty = detector.detect(" \n\t")
    short = detector.detect("ab")
    assert empty.reason is AbstentionReason.EMPTY_TEXT
    assert short.reason is AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT
    assert not model.calls


def test_confidence_and_margin_abstention(tmp_path: Path) -> None:
    low_confidence = FakeFastTextModel(
        ("__label__en", "__label__tr"),
        (0.60, 0.10),
    )
    low_margin = FakeFastTextModel(
        ("__label__en", "__label__tr"),
        (0.90, 0.80),
    )
    assert (
        _detector(tmp_path, low_confidence).detect("valid text").reason
        is AbstentionReason.LOW_CONFIDENCE
    )
    assert (
        _detector(tmp_path, low_margin).detect("valid text").reason
        is AbstentionReason.LOW_MARGIN
    )


def test_unknown_model_labels_abstain(tmp_path: Path) -> None:
    model = FakeFastTextModel(("__label__de",), (0.99,))
    result = _detector(tmp_path, model).detect("valid text")
    assert result.language is LanguageCode.UND
    assert result.reason is AbstentionReason.NO_SUPPORTED_PREDICTIONS


def test_missing_model_fails_before_optional_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model not found"):
        FastTextLanguageDetector(tmp_path / "missing.bin")


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(tmp_path: Path, top_k) -> None:
    model = FakeFastTextModel(("__label__en",), (0.99,))
    with pytest.raises(ValueError):
        _detector(tmp_path, model).detect("valid text", top_k=top_k)

import pytest

from kurdish_nlp.langid.schemas import (
    AbstentionReason,
    DetectionResult,
    LanguageCode,
    Prediction,
)


def test_language_codes_include_runtime_und() -> None:
    assert LanguageCode("ckb") is LanguageCode.CKB
    assert LanguageCode("und") is LanguageCode.UND
    with pytest.raises(ValueError):
        LanguageCode("ku")


def test_detection_result_creation_and_serialization() -> None:
    alternatives = (
        Prediction("ckb", 0.94),
        Prediction("fa", 0.04),
        Prediction("ar", 0.02),
    )
    result = DetectionResult(
        language="ckb",
        confidence=0.94,
        is_reliable=True,
        alternatives=alternatives,
        margin=0.90,
        reason=None,
        model_identifier="fixture",
        model_version="1",
    )

    assert result.language is LanguageCode.CKB
    assert result.to_dict()["alternatives"][1]["language"] == "fa"


def test_unreliable_result_requires_und_and_reason() -> None:
    with pytest.raises(ValueError, match="must be und"):
        DetectionResult(
            language="ckb",
            confidence=0.0,
            is_reliable=False,
            alternatives=(),
            margin=0.0,
            reason=AbstentionReason.EMPTY_TEXT,
            model_identifier="fixture",
            model_version="1",
        )


def test_scores_must_be_finite_and_bounded() -> None:
    with pytest.raises(ValueError):
        Prediction("ckb", 1.1)
    with pytest.raises(ValueError):
        Prediction("ckb", float("nan"))

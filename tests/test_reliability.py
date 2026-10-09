from kurdish_nlp.langid.reliability import ReliabilityPolicy
from kurdish_nlp.langid.schemas import AbstentionReason, LanguageCode, Prediction

POLICY = ReliabilityPolicy(
    confidence_threshold=0.70,
    margin_threshold=0.20,
    minimum_letter_count=3,
)


def test_reliable_prediction() -> None:
    result = POLICY.evaluate(
        "دەقێکی کوردی",
        (Prediction("ckb", 0.90), Prediction("fa", 0.10)),
    )
    assert result.language is LanguageCode.CKB
    assert result.is_reliable
    assert result.margin == 0.80


def test_empty_and_short_text_abstain() -> None:
    empty = POLICY.evaluate("", ())
    short = POLICY.evaluate("ab", ())
    assert empty.reason is AbstentionReason.EMPTY_TEXT
    assert short.reason is AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT


def test_low_confidence_abstains_before_margin() -> None:
    result = POLICY.evaluate(
        "valid text",
        (Prediction("en", 0.60), Prediction("tr", 0.10)),
    )
    assert result.language is LanguageCode.UND
    assert result.reason is AbstentionReason.LOW_CONFIDENCE


def test_small_top1_top2_margin_abstains() -> None:
    result = POLICY.evaluate(
        "valid text",
        (Prediction("en", 0.80), Prediction("tr", 0.70)),
    )
    assert result.reason is AbstentionReason.LOW_MARGIN
    assert result.margin == pytest.approx(0.10)


import pytest

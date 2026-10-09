from kurdish_nlp.langid.normalization import lexical_letter_count, normalize_text


def test_unicode_and_whitespace_normalization() -> None:
    assert normalize_text("  cafe\u0301\n\t test  ") == "café test"


def test_control_and_invisible_characters_are_removed() -> None:
    assert normalize_text("\ufeffabc\u202edef\x00") == "abcdef"


def test_distinguishing_arabic_persian_and_kurdish_characters_are_preserved() -> None:
    text = "ک ك ی ي ە ڕ ۆ ێ ڵ پ چ ژ گ"
    assert normalize_text(text) == text


def test_join_controls_are_preserved() -> None:
    assert normalize_text("می\u200cروم کورد\u200dی") == "می\u200cروم کورد\u200dی"


def test_lexical_count_is_script_agnostic() -> None:
    assert lexical_letter_count("123 کورد! abc") == 7

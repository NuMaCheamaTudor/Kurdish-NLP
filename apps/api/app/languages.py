"""Display metadata for the runtime language codes.

This is the single place user-facing language names are defined; the web client
renders whatever the API returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from kurdish_nlp.langid import LanguageCode


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    code: str
    name: str
    alt_name: str | None
    kurdish: bool
    script: str
    direction: str


LANGUAGES: dict[str, LanguageInfo] = {
    info.code: info
    for info in (
        LanguageInfo("ckb", "Sorani", "Central Kurdish", True, "Arabic", "rtl"),
        LanguageInfo("kmr", "Kurmanji", "Northern Kurdish", True, "Latin", "ltr"),
        LanguageInfo("sdh", "Southern Kurdish", None, True, "Arabic", "rtl"),
        LanguageInfo("ar", "Arabic", None, False, "Arabic", "rtl"),
        LanguageInfo("fa", "Persian", None, False, "Arabic", "rtl"),
        LanguageInfo("tr", "Turkish", None, False, "Latin", "ltr"),
        LanguageInfo("en", "English", None, False, "Latin", "ltr"),
    )
}

UNDETERMINED = LanguageInfo("und", "Inconclusive", None, False, "", "auto")

SUPPORTED_CODES: tuple[str, ...] = tuple(LANGUAGES)

assert set(SUPPORTED_CODES) == {
    code.value for code in LanguageCode if code is not LanguageCode.UND
}, "language metadata must cover exactly the trainable codes"


def language_info(code: str) -> LanguageInfo:
    return LANGUAGES.get(code, UNDETERMINED)

"""Conservative text normalization for language identification.

This module intentionally does not map Arabic, Persian, Sorani, or Southern
Kurdish letters into one another. Code points such as Arabic/Persian yeh and kaf,
and Kurdish-specific letters, can be valuable language-identification evidence.

Zero-width non-joiner and joiner are preserved because they can encode genuine
orthographic choices. Other format/control characters (BOM, bidi overrides,
zero-width space, and similar invisible formatting) are removed. All whitespace
is reduced to one ASCII space after Unicode NFC normalization.
"""

from __future__ import annotations

import unicodedata

_PRESERVED_FORMAT_CHARACTERS = frozenset({"\u200c", "\u200d"})


def normalize_text(text: str) -> str:
    """Return conservatively normalized text without language-specific folding."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)
    output: list[str] = []
    previous_was_space = False

    for character in normalized:
        if character.isspace():
            if output and not previous_was_space:
                output.append(" ")
            previous_was_space = True
            continue

        category = unicodedata.category(character)
        if category == "Cf" and character not in _PRESERVED_FORMAT_CHARACTERS:
            continue
        if category in {"Cc", "Cs"}:
            continue

        output.append(character)
        previous_was_space = False

    return "".join(output).strip()


def lexical_letter_count(text: str) -> int:
    """Count Unicode letters without assuming a particular script."""

    return sum(unicodedata.category(character).startswith("L") for character in text)

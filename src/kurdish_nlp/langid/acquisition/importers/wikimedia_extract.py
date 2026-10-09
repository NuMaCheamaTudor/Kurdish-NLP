"""Conservative, offline extraction of prose from MediaWiki revision wikitext."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from kurdish_nlp.langid.normalization import lexical_letter_count, normalize_text

EXTRACTION_VERSION = "4"
_HEADING = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$")
_NON_PROSE_SECTION = re.compile(
    r"^(references|notes|sources|bibliography|external links|further reading|"
    r"سەرچاوەکان|پەیوەندییە دەرەکییەکان|ژێدەرەکان|المراجع|المصادر|"
    r"وصلات خارجية|منابع|پیوندهای بیرونی|kaynakça|kaynaklar|dış bağlantılar)$",
    re.IGNORECASE,
)
_REDIRECT = re.compile(
    r"^\s*#\s*(?:redirect|گۆڕانەوە|تحويل|تغییرمسیر|yönlendir)", re.IGNORECASE
)
_DISAMBIGUATION = re.compile(
    r"\{\{\s*(?:disambig|dab|ابهام.?زدایی|ابهام|ڕوونکردنەوە|anlam ayrımı)",
    re.IGNORECASE,
)
_REMOVE_TAG_BLOCKS = re.compile(
    r"<(ref|math|gallery|syntaxhighlight|source|code|pre|timeline|chem|score)"
    r"\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SELF_CLOSING_REF = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CITATION = re.compile(r"\[(?:\d+|citation needed)\]", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?؟])\s+")
_RESIDUAL_MARKUP = re.compile(
    r"\[\[|\]\]|\{\{|\}\}|\||'{2,}|<[^>]*>|(?:^|\s)(?:file|image):",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    original: str
    section: str | None
    index: int
    flags: tuple[str, ...]


def script_flags(text: str, label: str) -> tuple[str, ...]:
    """Diagnostic flags only; script is never treated as a ground-truth label."""
    scripts: set[str] = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if name.startswith("ARABIC"):
            scripts.add("arabic")
        elif name.startswith("LATIN"):
            scripts.add("latin")
        else:
            scripts.add("other")
    flags: list[str] = []
    if len(scripts) > 1:
        flags.append("mixed_script")
    expected = "latin" if label in {"kmr", "tr", "en"} else "arabic"
    if expected not in scripts and scripts:
        flags.append("unexpected_script")
    if not scripts:
        flags.append("no_letters")
    return tuple(flags)


def _clean_wikitext(fragment: str) -> str:
    try:
        import mwparserfromhell
    except ImportError as error:
        raise RuntimeError(
            "Wikipedia extraction requires pip install '.[wikimedia]'"
        ) from error

    fragment = _COMMENT.sub("", fragment)
    fragment = _REMOVE_TAG_BLOCKS.sub("", fragment)
    fragment = _SELF_CLOSING_REF.sub("", fragment)
    code = mwparserfromhell.parse(fragment)
    for link in code.filter_wikilinks(recursive=True):
        target = str(link.title).strip().lower()
        if target.startswith(
            (
                "file:",
                "image:",
                "category:",
                "وێنە:",
                "پەڕگە:",
                "پرونده:",
                "تصویر:",
                "رده:",
                "ملف:",
                "صورة:",
                "تصنيف:",
                "dosya:",
                "kategori:",
            )
        ):
            try:
                code.remove(link)
            except ValueError:
                pass
    result = html.unescape(code.strip_code(normalize=True, collapse=True))
    result = _URL.sub(" ", result)
    result = _CITATION.sub("", result)
    return result.strip()


def _outside_templates(line: str, depth: int) -> tuple[str, int]:
    """Drop nested templates across line boundaries, retaining adjacent prose."""
    output: list[str] = []
    position = 0
    while position < len(line):
        pair = line[position : position + 2]
        if pair == "{{":
            depth += 1
            position += 2
        elif pair == "}}" and depth:
            depth -= 1
            position += 2
        else:
            if depth == 0:
                output.append(line[position])
            position += 1
    return "".join(output), depth


def _prose_paragraphs(wikitext: str) -> list[tuple[str, str | None]]:
    wikitext = _SELF_CLOSING_REF.sub(
        "", _REMOVE_TAG_BLOCKS.sub("", _COMMENT.sub("", wikitext))
    )
    paragraphs: list[tuple[str, str | None]] = []
    pending: list[str] = []
    section: str | None = None
    skip_section = False
    in_table = False
    template_depth = 0

    def flush() -> None:
        if pending and not skip_section:
            cleaned = _clean_wikitext(" ".join(pending))
            if cleaned:
                paragraphs.append((cleaned, section))
        pending.clear()

    for raw_line in wikitext.splitlines():
        line, template_depth = _outside_templates(raw_line, template_depth)
        line = line.strip()
        if line.startswith("{|"):
            flush()
            in_table = True
        if in_table:
            if line.endswith("|}"):
                in_table = False
            continue
        heading = _HEADING.fullmatch(line)
        if heading:
            flush()
            section = _clean_wikitext(heading.group(1)) or None
            skip_section = bool(section and _NON_PROSE_SECTION.fullmatch(section))
            continue
        if not line:
            flush()
            continue
        if line.startswith(
            ("*", "#", ";", ":", "|", "!", "[[Category:", "[[رده:", "[[تصنيف:")
        ):
            flush()
            continue
        if skip_section:
            continue
        pending.append(line)
    flush()
    return paragraphs


def extract_segments(
    wikitext: str,
    *,
    label: str,
    mode: str = "bounded",
    max_tokens: int = 80,
    max_segments: int = 30,
    rejections: Counter[str] | None = None,
) -> tuple[list[Segment], str | None]:
    """Return prose segments and a page-level rejection reason, if any."""
    if mode not in {"sentence", "paragraph", "bounded"}:
        raise ValueError("segment mode must be sentence, paragraph, or bounded")
    if max_tokens < 5 or max_segments < 1:
        raise ValueError("max_tokens must be >= 5 and max_segments >= 1")
    if not wikitext.strip():
        return [], "empty"
    if _REDIRECT.match(wikitext):
        return [], "redirect"
    if _DISAMBIGUATION.search(wikitext[:3000]):
        return [], "disambiguation"

    output: list[Segment] = []
    for paragraph, section in _prose_paragraphs(wikitext):
        sentences = [
            item.strip() for item in _SENTENCE.split(paragraph) if item.strip()
        ]
        if mode == "sentence":
            chunks = sentences
        elif mode == "paragraph":
            chunks = [paragraph]
        else:
            chunks = []
            current: list[str] = []
            current_tokens = 0
            for sentence in sentences:
                count = len(sentence.split())
                if current and current_tokens + count > max_tokens:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                current.append(sentence)
                current_tokens += count
            if current:
                chunks.append(" ".join(current))
        for raw in chunks:
            text = normalize_text(raw)
            if len(text.split()) < 2 or lexical_letter_count(text) < 8:
                if rejections is not None:
                    rejections["insufficient_lexical_content"] += 1
                continue
            if len(text.split()) > max_tokens:
                if rejections is not None:
                    rejections["over_max_tokens"] += 1
                continue
            if _RESIDUAL_MARKUP.search(text):
                if rejections is not None:
                    rejections["residual_markup"] += 1
                continue
            output.append(
                Segment(
                    text=text,
                    original=raw,
                    section=section,
                    index=len(output),
                    flags=script_flags(text, label),
                )
            )
            if len(output) >= max_segments:
                return output, None
    return output, None if output else "no_prose"

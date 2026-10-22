"""Strict BIO parsing and measurement. Every nonempty segment remains traceable."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LABEL = re.compile(r"[BI]-[A-Z][A-Z_]*\Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_bio(path: Path) -> list[dict]:
    records, rows = [], []
    source_sha = sha256(path)

    def flush():
        if not rows:
            return
        tokens, tags, row_lines, malformed = [], [], [], []
        for line, raw in rows:
            fields = raw.split()
            if len(fields) != 2:
                malformed.append({"line": line, "raw": raw, "fields": len(fields)})
            else:
                tokens.append(fields[0])
                tags.append(fields[1])
                row_lines.append(line)
        text = " ".join(tokens)
        records.append(
            {
                "id": f"seg-{len(records) + 1:06d}",
                "source_sha256": source_sha,
                "line_start": rows[0][0],
                "line_end": rows[-1][0],
                "row_lines": row_lines,
                "tokens": tokens,
                "tags": tags,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "malformed": malformed,
            }
        )
        rows.clear()

    # utf-8 strict: decoding failure is fatal, never replace undecodable bytes.
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            rows.append((line, raw))
        else:
            flush()
    flush()
    return records


def bio_errors(tags: list[str]) -> list[dict]:
    errors, previous = [], "O"
    for i, tag in enumerate(tags):
        if tag != "O" and LABEL.fullmatch(tag) is None:
            errors.append({"index": i, "kind": "invalid_label", "tag": tag})
        elif tag.startswith("I-") and previous not in ("B-" + tag[2:], tag):
            errors.append(
                {
                    "index": i,
                    "kind": "invalid_transition",
                    "tag": tag,
                    "previous": previous,
                }
            )
        previous = tag
    return errors


def spans(tags: list[str]) -> list[tuple[int, int, str]]:
    errors = bio_errors(tags)
    if errors:
        raise ValueError(f"Invalid BIO: {errors[:3]}")
    entities = []
    start, label = None, None
    for i, tag in enumerate([*tags, "O"]):
        if start is not None and tag != "I-" + label:
            entities.append((start, i, label))
            start, label = None, None
        if tag.startswith("B-"):
            start, label = i, tag[2:]
    return entities


def audit(records: list[dict], long_limit: int = 300) -> tuple[dict, list[dict]]:
    tag_counts, characters, features = Counter(), Counter(), Counter()
    starts, entity_tokens, text_groups, issues = (
        Counter(),
        Counter(),
        defaultdict(list),
        [],
    )
    lengths, prefixes, suffixes = [], defaultdict(list), defaultdict(list)
    suspicious_tokens = Counter()
    for r in records:
        n = len(r["tokens"])
        lengths.append(n)
        text_groups[tuple(r["tokens"])].append(r)
        tag_counts.update(r["tags"])
        errors = bio_errors(r["tags"])
        for error in errors:
            issues.append(
                dict(record=r["id"], line=r["row_lines"][error["index"]], **error)
            )
        for malformed in r["malformed"]:
            issues.append(dict(record=r["id"], kind="malformed_row", **malformed))
        for tag in r["tags"]:
            if LABEL.fullmatch(tag):
                entity_tokens[tag[2:]] += 1
                if tag.startswith("B-"):
                    starts[tag[2:]] += 1
        text = " ".join(r["tokens"])
        characters.update(text)
        flags = set()
        for word in r["tokens"]:
            if unicodedata.normalize("NFC", word) != word:
                flags.add("non_nfc")
            if unicodedata.normalize("NFKC", word) != word:
                flags.add("nfkc_changes")
            if any(unicodedata.category(c) == "Cf" for c in word):
                flags.add("format_controls")
            if any(
                "ARABIC" in unicodedata.name(c, "")
                and "PRESENTATION" in unicodedata.name(c, "")
                for c in word
            ):
                flags.add("arabic_presentation_forms")
            if any(c in word for c in "يكى"):
                flags.add("arabic_yeh_kaf_variants")
            if any(c.isdigit() for c in word):
                flags.add("digits")
            if any("LATIN" in unicodedata.name(c, "") for c in word):
                flags.add("latin")
            if len(word) > 50:
                flags.add("token_over_50_chars")
                suspicious_tokens[word] += 1
            if re.search(r"https?://|www\.|@", word, re.IGNORECASE):
                flags.add("url_or_email_candidate")
            if re.search(r"\w[،؛:!?؟.,]\w", word):
                flags.add("internal_punctuation_candidate")
        if n > long_limit:
            flags.add("over_300_tokens_possible_concatenation")
        for flag in flags:
            features[flag] += 1
        if flags:
            issues.append(
                {"record": r["id"], "kind": "record_flags", "flags": sorted(flags)}
            )
        if n >= 5:
            prefixes[tuple(r["tokens"][:5])].append(r["id"])
            suffixes[tuple(r["tokens"][-5:])].append(r["id"])
    duplicate_groups = [rs for rs in text_groups.values() if len(rs) > 1]
    conflicts = [
        rs for rs in duplicate_groups if len({tuple(r["tags"]) for r in rs}) > 1
    ]
    for rs in duplicate_groups:
        issues.append(
            {
                "kind": "exact_duplicate_group",
                "records": [r["id"] for r in rs],
                "conflicting": len({tuple(r["tags"]) for r in rs}) > 1,
            }
        )
    ordered_lengths = sorted(lengths)
    quantiles = (
        {
            str(p): ordered_lengths[int((len(lengths) - 1) * p / 100)]
            for p in (0, 25, 50, 75, 90, 95, 99, 100)
        }
        if lengths
        else {}
    )
    special = {
        f"U+{ord(c):04X} {unicodedata.name(c, 'UNNAMED')}": count
        for c, count in sorted(characters.items())
        if unicodedata.category(c) in {"Cf", "Mn", "Cc"} or c in "يكىیكکـ"
    }
    boilerplate = []
    for kind, values in (("prefix", prefixes), ("suffix", suffixes)):
        for phrase, ids in sorted(values.items(), key=lambda x: (-len(x[1]), x[0]))[
            :30
        ]:
            if len(ids) >= 5:
                boilerplate.append(
                    {
                        "kind": kind,
                        "phrase": " ".join(phrase),
                        "count": len(ids),
                        "records": ids,
                    }
                )
    return {
        "segments": len(records),
        "parsed_token_rows": sum(lengths),
        "valid_bio_token_rows": sum(
            v for t, v in tag_counts.items() if t == "O" or LABEL.fullmatch(t)
        ),
        "malformed_rows": sum(len(r["malformed"]) for r in records),
        "raw_tag_frequencies": dict(sorted(tag_counts.items())),
        "entity_labels": sorted(entity_tokens),
        "entity_token_frequencies": dict(sorted(entity_tokens.items())),
        "explicit_B_entity_starts": dict(sorted(starts.items())),
        "entity_count_caveat": "Raw B counts only; invalid I transitions are not silently repaired into entities.",
        "repeated_exact_records": sum(len(rs) - 1 for rs in duplicate_groups),
        "exact_duplicate_groups": len(duplicate_groups),
        "conflicting_text_groups": len(conflicts),
        "records_in_conflicting_groups": sum(map(len, conflicts)),
        "invalid_labels": sum(i["kind"] == "invalid_label" for i in issues),
        "invalid_bio_transitions": sum(
            i["kind"] == "invalid_transition" for i in issues
        ),
        "records_with_bio_errors": sum(bool(bio_errors(r["tags"])) for r in records),
        "length_quantiles_tokens": quantiles,
        "segments_over_300_tokens": sum(n > long_limit for n in lengths),
        "unicode_record_features": dict(sorted(features.items())),
        "unicode_special_characters": special,
        "arabic_named_characters": sum(
            v for c, v in characters.items() if "ARABIC" in unicodedata.name(c, "")
        ),
        "latin_named_characters": sum(
            v for c, v in characters.items() if "LATIN" in unicodedata.name(c, "")
        ),
        "long_token_examples": suspicious_tokens.most_common(30),
        "potential_boilerplate": boilerplate,
        "metadata": {
            "verified_document_ids": False,
            "verified_source_ids": False,
            "original_whitespace": False,
            "license_verified": False,
            "unit": "blank-line-delimited segment, not an independent document",
        },
    }, issues

"""Read-only streaming quality and PARME-overlap audit for an acquired snapshot."""

from __future__ import annotations

import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from kurdish_nlp.langid.normalization import normalize_text

ROOT = Path(__file__).resolve().parents[1]
TATOEBA = ROOT / "data/langid/processed/tatoeba.jsonl"
PROVENANCE = ROOT / "data/langid/processed/tatoeba.provenance.jsonl"
PARME = ROOT / "data/langid/processed/parme.sdh.jsonl"
PARME_PROVENANCE = ROOT / "data/langid/processed/parme.provenance.jsonl"
KURDISH = frozenset({"ckb", "kmr", "sdh"})


def _script(text: str) -> tuple[bool, bool, bool]:
    arabic = latin = other = False
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        arabic |= name.startswith("ARABIC")
        latin |= name.startswith("LATIN")
        other |= not name.startswith(("ARABIC", "LATIN"))
    return arabic, latin, other


def main() -> None:
    parme_rows = [json.loads(line) for line in PARME.open(encoding="utf-8")]
    parme_sidecars = [
        json.loads(line) for line in PARME_PROVENANCE.open(encoding="utf-8")
    ]
    parme_texts: dict[str, list[dict]] = defaultdict(list)
    for row in parme_rows:
        parme_texts[normalize_text(row["text"])].append(row)
    parme_original_hashes = {
        row["original_text_sha256"]
        for row in parme_sidecars
        if row.get("original_text_sha256")
    }
    contributors: dict[str, Counter[str]] = defaultdict(Counter)
    scripts: dict[str, Counter[str]] = defaultdict(Counter)
    orthography: dict[str, Counter[str]] = defaultdict(Counter)
    suspicious: dict[str, list[dict]] = defaultdict(list)
    cross_source_normalized: list[dict] = []
    cross_source_exact_original = 0
    sdh_lengths: list[int] = []
    count = 0
    with (
        TATOEBA.open(encoding="utf-8") as canonical,
        PROVENANCE.open(encoding="utf-8") as provenance,
    ):
        for count, (canonical_line, provenance_line) in enumerate(
            zip_longest(canonical, provenance), 1
        ):
            if canonical_line is None or provenance_line is None:
                raise ValueError(
                    f"Tatoeba canonical/provenance length mismatch at {count}"
                )
            row = json.loads(canonical_line)
            sidecar = json.loads(provenance_line)
            if row["id"] != sidecar["canonical_record_id"]:
                raise ValueError(f"Tatoeba canonical/provenance ID mismatch at {count}")
            label = row["label"]
            contributors[label][sidecar.get("contributor_id") or "<missing>"] += 1
            if label not in KURDISH:
                continue
            text = row["text"]
            arabic, latin, other = _script(text)
            scripts[label]["arabic_letters"] += arabic
            scripts[label]["latin_letters"] += latin
            scripts[label]["other_script_letters"] += other
            scripts[label]["mixed_arabic_latin"] += arabic and latin
            if arabic and latin and len(suspicious[label]) < 12:
                suspicious[label].append(
                    {
                        "id": row["id"],
                        "text": text[:120],
                        "reason": "mixed_arabic_latin",
                    }
                )
            expected = latin if label == "kmr" else arabic
            unexpected = arabic if label == "kmr" else latin
            if not expected and unexpected:
                scripts[label]["unexpected_script_only"] += 1
                if len(suspicious[label]) < 12:
                    suspicious[label].append(
                        {
                            "id": row["id"],
                            "text": text[:120],
                            "reason": "unexpected_script_only",
                        }
                    )
            if not arabic and not latin and not other:
                scripts[label]["no_letters"] += 1
            for mark in ("ي", "ی", "ى", "ك", "ک", "ە", "ه", "ڕ", "\u200c"):
                orthography[label][mark] += text.count(mark)
            if label == "sdh":
                sdh_lengths.append(len(text.split()))
                matches = parme_texts.get(normalize_text(text), [])
                for parme in matches:
                    cross_source_normalized.append(
                        {
                            "tatoeba_id": row["id"],
                            "tatoeba_split": row["split"],
                            "parme_id": parme["id"],
                            "parme_split": parme["split"],
                            "same_split": row["split"] == parme["split"],
                        }
                    )
                cross_source_exact_original += (
                    sidecar.get("original_text_sha256") in parme_original_hashes
                )
    result = {
        "tatoeba_records_scanned": count,
        "parme_sdh_records": len(parme_rows),
        "parme_sdh_token_lengths": {
            "mean": statistics.mean(len(row["text"].split()) for row in parme_rows),
            "median": statistics.median(len(row["text"].split()) for row in parme_rows),
        },
        "tatoeba_sdh_token_lengths": {
            "mean": statistics.mean(sdh_lengths),
            "median": statistics.median(sdh_lengths),
        },
        "cross_source_exact_original_tatoeba_rows": cross_source_exact_original,
        "cross_source_normalized_text_pairs": len(cross_source_normalized),
        "cross_source_split_conflict_pairs": sum(
            not item["same_split"] for item in cross_source_normalized
        ),
        "cross_source_matches": cross_source_normalized[:25],
        "contributor_top_10_by_language": {
            label: counts.most_common(10)
            for label, counts in sorted(contributors.items())
        },
        "kurdish_script_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(scripts.items())
        },
        "kurdish_orthography_character_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(orthography.items())
        },
        "suspicious_kurdish_examples_for_review": dict(sorted(suspicious.items())),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

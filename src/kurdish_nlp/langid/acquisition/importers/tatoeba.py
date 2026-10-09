"""Generic Tatoeba importer. Network access is confined to ``acquire``.

The official exports are mutable. A local snapshot receipt pins the exact three
archives used together; import refuses unreceipted or changed files.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any
from urllib.request import Request, urlopen

from kurdish_nlp.langid.acquisition.licenses import LicenseId, LicenseSpec
from kurdish_nlp.langid.acquisition.manifests import SourceManifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import (
    ProvenanceRecord,
    Transformation,
    save_sidecars,
    validate_linkage,
)
from kurdish_nlp.langid.acquisition.schemas import strict_json_loads
from kurdish_nlp.langid.dataset import DatasetRecord, validate_jsonl
from kurdish_nlp.langid.normalization import normalize_text

DEFAULT_MANIFEST = Path("configs/langid/sources/tatoeba.v1.json")
DEFAULT_RAW_DIR = Path("data/langid/raw/tatoeba")
DEFAULT_OUTPUT_DIR = Path("data/langid/processed")
EXPORT_ROOT = "https://downloads.tatoeba.org/exports/"
FILES = ("sentences_detailed.tar.bz2", "sentences_CC0.tar.bz2", "links.tar.bz2")
RECEIPT_NAME = "tatoeba.snapshot.json"
LARGE_ARCHIVE_THRESHOLD = 50 * 1024 * 1024
LANGUAGE_MAP = {
    "ckb": "ckb",
    "kmr": "kmr",
    "sdh": "sdh",
    "ara": "ar",
    "pes": "fa",
    "tur": "tr",
    "eng": "en",
}
TARGETS = frozenset(LANGUAGE_MAP.values())
LICENSE_URLS = {
    LicenseId.CC0: "https://creativecommons.org/publicdomain/zero/1.0/",
    LicenseId.CC_BY_2: "https://creativecommons.org/licenses/by/2.0/",
    LicenseId.CC_BY_2_FR: "https://creativecommons.org/licenses/by/2.0/fr/",
    LicenseId.CC_BY_4: "https://creativecommons.org/licenses/by/4.0/",
    LicenseId.CC_BY_SA_4: "https://creativecommons.org/licenses/by-sa/4.0/",
    LicenseId.CC_BY_ND_4: "https://creativecommons.org/licenses/by-nd/4.0/",
    LicenseId.CC_BY_NC_SA_4: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    LicenseId.CC_BY_NC_ND_4: "https://creativecommons.org/licenses/by-nc-nd/4.0/",
}
LICENSE_ALIASES = {
    "CC0": LicenseId.CC0,
    "CC0-1.0": LicenseId.CC0,
    "CC-BY-2.0": LicenseId.CC_BY_2,
    "CC BY 2.0 FR": LicenseId.CC_BY_2_FR,
    "CC-BY-2.0-FR": LicenseId.CC_BY_2_FR,
    "CC-BY-4.0": LicenseId.CC_BY_4,
    "CC-BY-SA-4.0": LicenseId.CC_BY_SA_4,
    "CC-BY-ND-4.0": LicenseId.CC_BY_ND_4,
    "CC-BY-NC-SA-4.0": LicenseId.CC_BY_NC_SA_4,
    "CC-BY-NC-ND-4.0": LicenseId.CC_BY_NC_ND_4,
}
_CONTENT = re.compile(r"[^\W\d_]", re.UNICODE)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_manifest(manifest: SourceManifest) -> None:
    if manifest.source_id != "tatoeba" or manifest.source_url != EXPORT_ROOT:
        raise ValueError("Tatoeba importer requires the reviewed official export URL")
    if manifest.license.identifier is not LicenseId.CC_BY_2_FR:
        raise ValueError(
            "Tatoeba manifest must declare the official default text license"
        )


def _languages(languages: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(sorted(set(languages)))
    if not selected or any(item not in TARGETS for item in selected):
        raise ValueError(f"languages must be a nonempty subset of {sorted(TARGETS)}")
    return selected


def plan_acquisition(
    manifest: SourceManifest,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
) -> dict[str, Any]:
    _check_manifest(manifest)
    decision = evaluate(manifest, role="training", profile=profile)
    return {
        "source_id": "tatoeba",
        "mutable_upstream": True,
        "snapshot_strategy": "pin per-file SHA-256 in ignored local receipt; retain archives",
        "files": [
            {"url": EXPORT_ROOT + name, "destination": str(raw_dir / name)}
            for name in FILES
        ],
        "receipt": str(raw_dir / RECEIPT_NAME),
        "policy": decision.to_dict(),
    }


def _receipt(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / RECEIPT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Tatoeba snapshot receipt missing: {path}; run acquire tatoeba"
        )
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("files"), dict)
        or value.get("schema_version") != 1
        or value.get("source_id") not in (None, "tatoeba")
        or set(value.get("files", {})) != set(FILES)
    ):
        raise ValueError("invalid Tatoeba snapshot receipt")
    for name in FILES:
        item = value["files"][name]
        if not isinstance(item, dict):
            raise TypeError(f"invalid Tatoeba receipt entry: {name}")
        if item.get("url") != EXPORT_ROOT + name:
            raise ValueError(f"unexpected Tatoeba URL in receipt: {name}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid SHA-256 in receipt: {name}")
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Tatoeba raw file missing: {path}")
        if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Tatoeba size mismatch for {name}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Tatoeba checksum mismatch for {name}: {actual} != {expected}"
            )
    digest = _hash(
        json.dumps(
            {name: value["files"][name]["sha256"] for name in FILES},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if value.get("snapshot_id") != digest:
        raise ValueError("Tatoeba snapshot ID does not match file checksums")
    return value


def acquire(
    manifest: SourceManifest,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
    opener: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Explicitly fetch a rolling export, then freeze all three file hashes."""
    _check_manifest(manifest)
    decision = evaluate(manifest, role="training", profile=profile)
    if not decision.allowed:
        raise ValueError(
            f"Tatoeba acquisition blocked: {decision.to_dict()['reasons']}"
        )
    if (raw_dir / RECEIPT_NAME).exists():
        return _receipt(raw_dir), True
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any((raw_dir / name).exists() for name in FILES):
        raise ValueError("unreceipted Tatoeba raw files exist; do not overwrite them")
    open_url = opener or urlopen
    files: dict[str, dict[str, Any]] = {}
    for name in FILES:
        partial = raw_dir / (name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "kurdish-nlp-tatoeba-importer/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with open_url(
            Request(EXPORT_ROOT + name, headers=headers), timeout=120
        ) as response:
            if response.status not in (200, 206):
                raise ValueError(f"unexpected Tatoeba HTTP status: {response.status}")
            content_range = response.headers.get("Content-Range", "")
            if response.status == 206 and not re.fullmatch(
                rf"bytes {offset}-\d+/\d+", content_range
            ):
                raise ValueError("Tatoeba resume Content-Range does not match offset")
            mode = "ab" if response.status == 206 else "wb"
            with partial.open(mode) as target:
                shutil.copyfileobj(response, target)
            modified = response.headers.get("Last-Modified")
        checksum = _sha256(partial)
        files[name] = {
            "url": EXPORT_ROOT + name,
            "sha256": checksum,
            "bytes": partial.stat().st_size,
            "last_modified": modified,
        }
    for name in FILES:
        os.replace(raw_dir / (name + ".part"), raw_dir / name)
    snapshot_id = _hash(
        json.dumps(
            {name: files[name]["sha256"] for name in FILES},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    receipt = {
        "schema_version": 1,
        "source_id": "tatoeba",
        "manifest_version": manifest.manifest_version,
        "acquired_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_id,
        "files": files,
    }
    (raw_dir / RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return _receipt(raw_dir), False


def _archive_rows(path: Path, member_name: str) -> Iterable[list[str]]:
    try:
        with tarfile.open(path, "r:bz2") as archive:
            matches = [
                member
                for member in archive
                if member.isfile() and Path(member.name).name == member_name
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one {member_name} in {path}")
            binary = archive.extractfile(matches[0])
            if binary is None:
                raise ValueError(f"cannot extract {member_name}")
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as source:
                # Tatoeba's export is literal TSV, not RFC-style quoted CSV.
                # Quotation marks inside sentence text must remain literal.
                yield from csv.reader(
                    source, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True
                )
    except (tarfile.TarError, csv.Error, UnicodeError) as error:
        raise ValueError(f"invalid Tatoeba export {path}: {error}") from error


class TranslationGraph:
    """Compact integer disjoint-set forest; root is always minimum sentence ID."""

    def __init__(self) -> None:
        self.parent = array("I", [0])
        self.edge_count = 0

    def find(self, node: int) -> int:
        if node <= 0 or node >= 2**32:
            raise ValueError("sentence ID must be a positive 32-bit integer")
        parent = self.parent
        if node >= len(parent):
            parent.extend([0] * (node + 1 - len(parent)))
        if parent[node] == 0:
            parent[node] = node
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
        self.edge_count += 1

    def group_id(self, node: int) -> str:
        return f"tatoeba:translation-group:{self.find(node)}"


def _positive_id(value: str) -> int:
    if not value.isdecimal() or not 0 < int(value) < 2**32:
        raise ValueError("invalid Tatoeba sentence ID")
    return int(value)


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: int
    language_code: str
    label: str
    text: str
    username: str | None
    created_at: str | None
    modified_at: str | None
    license_raw: str
    license_source: str
    line: int


def parse_sentences(
    rows: Iterable[list[str]],
    cc0_ids: set[int],
    *,
    rejection_counts: Counter[str],
) -> list[Sentence]:
    result: list[Sentence] = []
    seen_ids: set[int] = set()
    for line, fields in enumerate(rows, 1):
        if len(fields) not in (6, 7):
            rejection_counts["malformed_sentence_columns"] += 1
            continue
        try:
            sentence_id = _positive_id(fields[0])
        except ValueError:
            rejection_counts["invalid_sentence_id"] += 1
            continue
        if sentence_id in seen_ids:
            rejection_counts["duplicate_sentence_id"] += 1
            continue
        seen_ids.add(sentence_id)
        code = fields[1]
        label = LANGUAGE_MAP.get(code)
        if label is None:
            rejection_counts["unsupported_language_code"] += 1
            continue
        if len(fields) == 7:
            license_raw, license_source = fields[6], "explicit_row_field"
        elif sentence_id in cc0_ids:
            license_raw, license_source = "CC0-1.0", "official_CC0_membership"
        else:
            license_raw, license_source = "CC-BY-2.0-FR", "official_export_default"
        result.append(
            Sentence(
                sentence_id,
                code,
                label,
                fields[2],
                fields[3] or None,
                fields[4] or None,
                fields[5] or None,
                license_raw,
                license_source,
                line,
            )
        )
    return result


def _cc0_ids(rows: Iterable[list[str]]) -> set[int]:
    ids: set[int] = set()
    for fields in rows:
        if len(fields) != 4:
            raise ValueError("malformed official CC0 export row")
        ids.add(_positive_id(fields[0]))
    return ids


def _license_decision(
    manifest: SourceManifest, raw: str, profile: str
) -> tuple[LicenseId | None, str | None]:
    identifier = LICENSE_ALIASES.get(raw)
    if identifier is None:
        return None, "missing_license" if not raw else "unknown_license"
    spec = LicenseSpec.for_identifier(identifier, LICENSE_URLS[identifier])
    decision = evaluate(
        replace(manifest, license=spec), role="training", profile=profile
    )
    if not decision.allowed:
        return None, "license_" + decision.reasons[0].code
    return identifier, None


def _check_ratios(ratios: tuple[float, float, float]) -> None:
    if (
        len(ratios) != 3
        or any(not 0 < ratio < 1 for ratio in ratios)
        or abs(sum(ratios) - 1) > 1e-9
    ):
        raise ValueError("train/dev/test ratios must be positive and sum to 1")


def split_for_group(
    group_id: str, seed: int, ratios: tuple[float, float, float]
) -> str:
    _check_ratios(ratios)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    score = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return (
        "train"
        if score < ratios[0]
        else "dev"
        if score < ratios[0] + ratios[1]
        else "test"
    )


def transform(
    sentences: list[Sentence],
    graph: TranslationGraph,
    manifest: SourceManifest,
    snapshot: dict[str, Any],
    *,
    languages: Iterable[str],
    profile: str = "commercial",
    max_per_language: int | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    cross_label_duplicates: str = "report",
    initial_rejections: Counter[str] | None = None,
    link_counts: Counter[int] | None = None,
    link_samples: dict[int, list[int]] | None = None,
) -> tuple[list[DatasetRecord], list[ProvenanceRecord], dict[str, Any]]:
    selected = _languages(languages)
    _check_ratios(ratios)
    if max_per_language is not None and max_per_language < 1:
        raise ValueError("max_per_language must be positive")
    if cross_label_duplicates not in ("report", "exclude"):
        raise ValueError("cross_label_duplicates must be report or exclude")
    rejected = Counter(initial_rejections or {})
    candidates: list[tuple[Sentence, str]] = []
    licenses: Counter[str] = Counter()
    license_values: Counter[str] = Counter()
    content_free_by_language: Counter[str] = Counter()
    content_free_kinds: Counter[str] = Counter()
    for row in sentences:
        if row.label not in selected:
            continue
        license_values[row.license_raw or "<missing>"] += 1
        license_id, reason = _license_decision(manifest, row.license_raw, profile)
        if reason:
            rejected[reason] += 1
            continue
        assert license_id is not None
        text = normalize_text(row.text)
        if not text or not _CONTENT.search(text):
            rejected["content_free"] += 1
            content_free_by_language[row.label] += 1
            content_free_kinds[
                "numeric_only"
                if any(char.isdigit() for char in text)
                else "punctuation_or_empty"
            ] += 1
            continue
        licenses[license_id.value] += 1
        candidates.append((row, text))
    before_sampling = Counter(row.label for row, _ in candidates)
    if max_per_language is not None:
        by_label_group: dict[str, dict[str, list[tuple[Sentence, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row, text in candidates:
            by_label_group[row.label][graph.group_id(row.sentence_id)].append(
                (row, text)
            )
        sampled: list[tuple[Sentence, str]] = []
        for label, groups in sorted(by_label_group.items()):
            count = 0
            for group_id in sorted(
                groups, key=lambda group: (_hash(f"{seed}:{group}"), group)
            ):
                entries = groups[group_id]
                if count + len(entries) <= max_per_language:
                    sampled.extend(entries)
                    count += len(entries)
                else:
                    rejected["sampled_out"] += len(entries)
        candidates = sampled
    after_sampling = Counter(row.label for row, _ in candidates)
    # The strict canonical validator disallows duplicate normalized text globally.
    # Keep the lowest stable sentence ID for each same-label duplicate, then
    # remove every cross-label collision while retaining its audit evidence.
    by_text: dict[str, list[tuple[Sentence, str]]] = defaultdict(list)
    for item in candidates:
        by_text[item[1]].append(item)
    accepted: list[tuple[Sentence, str]] = []
    cross_examples: list[dict[str, Any]] = []
    exact_duplicates = normalized_duplicates = cross_label_count = 0
    cross_label_exact_count = cross_label_normalized_only_count = 0
    for text, rows in sorted(by_text.items()):
        labels = {row.label for row, _ in rows}
        if len(labels) > 1:
            cross_label_count += 1
            by_original: dict[str, set[str]] = defaultdict(set)
            for row, _ in rows:
                by_original[row.text].add(row.label)
            if any(len(group_labels) > 1 for group_labels in by_original.values()):
                cross_label_exact_count += 1
            else:
                cross_label_normalized_only_count += 1
            rejected["cross_label_identical_text"] += len(rows)
            if cross_label_duplicates == "report" and len(cross_examples) < 25:
                cross_examples.append(
                    {
                        "text_sha256": _hash(text),
                        "labels": sorted(labels),
                        "sentence_ids": sorted(row.sentence_id for row, _ in rows),
                    }
                )
            continue
        winner = min(rows, key=lambda item: item[0].sentence_id)
        accepted.append(winner)
        for row, _ in rows:
            if row is winner[0]:
                continue
            if row.text == winner[0].text:
                exact_duplicates += 1
            else:
                normalized_duplicates += 1
            rejected["same_label_duplicate_text"] += 1
    accepted.sort(key=lambda item: (item[0].label, item[0].sentence_id))
    records: list[DatasetRecord] = []
    sidecars: list[ProvenanceRecord] = []
    group_sizes = Counter(graph.group_id(row.sentence_id) for row in sentences)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    naive_splits_by_group: dict[str, set[str]] = defaultdict(set)
    contributor_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    token_lengths: dict[str, list[int]] = defaultdict(list)
    normalization_changes: Counter[str] = Counter()
    short: dict[str, Counter[str]] = defaultdict(Counter)
    for row, text in accepted:
        group_id = graph.group_id(row.sentence_id)
        split = split_for_group(group_id, seed, ratios)
        record_id = f"tatoeba:{row.label}:{row.sentence_id}"
        license_id = LICENSE_ALIASES[row.license_raw]
        records.append(
            DatasetRecord.from_mapping(
                {
                    "id": record_id,
                    "text": text,
                    "label": row.label,
                    "source": f"tatoeba:{snapshot['snapshot_id']}",
                    "domain": "conversational_translation",
                    "document_id": group_id,
                    "license": license_id.value,
                    "split": split,
                }
            )
        )
        sidecars.append(
            ProvenanceRecord(
                schema_version=1,
                canonical_record_id=record_id,
                source_id="tatoeba",
                acquisition_manifest_version=manifest.manifest_version,
                original_record_id=str(row.sentence_id),
                alignment_group_id=group_id,
                original_url=f"https://tatoeba.org/en/sentences/show/{row.sentence_id}",
                source_file=FILES[0],
                source_row_number=row.line,
                contributor_id=row.username,
                original_language_label=row.language_code,
                original_license=row.license_raw,
                original_text_sha256=_hash(row.text),
                normalized_text_sha256=_hash(text),
                source_snapshot=snapshot["snapshot_id"],
                transformations=(
                    Transformation(
                        "normalize",
                        "1",
                        parameters={
                            "method": "conservative_nfc_whitespace",
                            "changed": text != row.text,
                        },
                    ),
                    Transformation(
                        "license_resolve",
                        "1",
                        parameters={
                            "source": row.license_source,
                            "license": license_id.value,
                            "created_at": row.created_at,
                            "modified_at": row.modified_at,
                        },
                    ),
                    Transformation(
                        "translation_component_split",
                        "1",
                        parameters={
                            "seed": seed,
                            "ratios": list(ratios),
                            "component_size_in_target_languages": group_sizes[group_id],
                            "direct_link_count": link_counts[row.sentence_id]
                            if link_counts is not None
                            else 0,
                            "linked_sentence_id_sample": link_samples.get(
                                row.sentence_id, []
                            )
                            if link_samples is not None
                            else [],
                        },
                    ),
                ),
            )
        )
        groups_by_split[group_id].add(split)
        naive_splits_by_group[group_id].add(
            split_for_group(f"tatoeba:sentence:{row.sentence_id}", seed, ratios)
        )
        contributor_by_split[row.username or "<missing>"][split] += 1
        length = len(text.split())
        token_lengths[row.label].append(length)
        short[row.label]["one_token"] += length == 1
        short[row.label]["one_to_five_tokens"] += length <= 5
        short[row.label]["url"] += bool(_URL.search(text))
        normalization_changes[row.label] += text != row.text
    if not records:
        raise ValueError("Tatoeba import has no usable records")
    validate_linkage(records, sidecars)
    group_language_sets: dict[str, set[str]] = defaultdict(set)
    for row in sentences:
        group_language_sets[graph.group_id(row.sentence_id)].add(row.label)
    contributor_counts = Counter(row.username or "<missing>" for row, _ in accepted)
    audit = {
        "source_id": "tatoeba",
        "snapshot_id": snapshot["snapshot_id"],
        "acquired_at": snapshot.get("acquired_at"),
        "source_last_modified_by_file": {
            name: snapshot["files"][name].get("last_modified") for name in FILES
        },
        "source_checksums": {name: snapshot["files"][name]["sha256"] for name in FILES},
        "languages": list(selected),
        "sentences_examined": len(sentences),
        "accepted_records": len(records),
        "rejected_counts": dict(sorted(rejected.items())),
        "rejected_records": sum(rejected.values()),
        "accepted_by_language": dict(
            sorted(Counter(record.label.value for record in records).items())
        ),
        "accepted_by_license": dict(
            sorted(Counter(record.license for record in records).items())
        ),
        "licensed_candidates_by_license": dict(sorted(licenses.items())),
        "candidate_license_values": dict(sorted(license_values.items())),
        "accepted_by_split": dict(
            sorted(Counter(record.split for record in records).items())
        ),
        "translation_component_count": len(group_language_sets),
        "component_size_in_target_languages": {
            "mean": sum(group_sizes.values()) / len(group_sizes),
            "median": median(group_sizes.values()),
        },
        "multilingual_target_components": sum(
            len(labels) > 1 for labels in group_language_sets.values()
        ),
        "naive_row_split_leakage_components": sum(
            len(splits) > 1 for splits in naive_splits_by_group.values()
        ),
        "component_leakage_count": sum(
            len(splits) > 1 for splits in groups_by_split.values()
        ),
        "contributor_counts": dict(sorted(contributor_counts.items())),
        "contributor_split_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(contributor_by_split.items())
        },
        "contributors_spanning_splits": sorted(
            name for name, splits in contributor_by_split.items() if len(splits) > 1
        ),
        "dominant_contributors_over_25_percent": {
            name: count
            for name, count in contributor_counts.items()
            if count / len(records) > 0.25
        },
        "token_lengths_by_language": {
            label: {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": median(lengths),
            }
            for label, lengths in sorted(token_lengths.items())
        },
        "short_text_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(short.items())
        },
        "content_free_rejections_by_language": dict(
            sorted(content_free_by_language.items())
        ),
        "content_free_rejection_kinds": dict(sorted(content_free_kinds.items())),
        "normalization_changes_by_language": dict(
            sorted(normalization_changes.items())
        ),
        "exact_same_label_duplicate_count": exact_duplicates,
        "normalized_same_label_duplicate_count": normalized_duplicates,
        "cross_label_identical_text_count": cross_label_count,
        "cross_label_exact_text_count": cross_label_exact_count,
        "cross_label_normalized_only_text_count": cross_label_normalized_only_count,
        "cross_label_examples": cross_examples,
        "sampling_before_by_language": dict(sorted(before_sampling.items())),
        "sampling_after_by_language": dict(sorted(after_sampling.items())),
        "max_per_language": max_per_language,
        "seed": seed,
        "split_ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "cross_label_duplicate_policy": cross_label_duplicates,
    }
    return records, sidecars, audit


def build(
    manifest: SourceManifest,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    languages: Iterable[str] = tuple(sorted(TARGETS)),
    profile: str = "commercial",
    max_per_language: int | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    cross_label_duplicates: str = "report",
) -> dict[str, Any]:
    """Verify the local snapshot, import it offline, and validate staged output."""
    _check_manifest(manifest)
    selected = _languages(languages)
    decision = evaluate(manifest, role="training", profile=profile)
    if not decision.allowed:
        raise ValueError(f"Tatoeba import blocked: {decision.to_dict()['reasons']}")
    snapshot = _receipt(raw_dir)
    if snapshot.get("manifest_version") not in (None, manifest.manifest_version):
        raise ValueError("Tatoeba receipt was created with another manifest version")
    if (
        raw_dir / FILES[0]
    ).stat().st_size >= LARGE_ARCHIVE_THRESHOLD and max_per_language is None:
        from kurdish_nlp.langid.acquisition.importers.tatoeba_large import build_large

        return build_large(
            manifest,
            snapshot,
            raw_dir=raw_dir,
            output_dir=output_dir,
            languages=selected,
            profile=profile,
            seed=seed,
            ratios=ratios,
            cross_label_duplicates=cross_label_duplicates,
        )
    cc0_ids = _cc0_ids(_archive_rows(raw_dir / FILES[1], "sentences_CC0.csv"))
    parse_rejections: Counter[str] = Counter()
    sentences = parse_sentences(
        _archive_rows(raw_dir / FILES[0], "sentences_detailed.csv"),
        cc0_ids,
        rejection_counts=parse_rejections,
    )
    skipped_other_languages = parse_rejections.pop("unsupported_language_code", 0)
    total_source_rows = (
        len(sentences) + skipped_other_languages + sum(parse_rejections.values())
    )
    target_ids = {row.sentence_id for row in sentences}
    link_counts: Counter[int] = Counter()
    link_samples: dict[int, list[int]] = defaultdict(list)
    graph = TranslationGraph()
    for fields in _archive_rows(raw_dir / FILES[2], "links.csv"):
        if len(fields) != 2:
            raise ValueError("malformed official links export row")
        left, right = _positive_id(fields[0]), _positive_id(fields[1])
        graph.union(left, right)
        if left in target_ids:
            link_counts[left] += 1
            if len(link_samples[left]) < 5:
                link_samples[left].append(right)
    records, sidecars, audit = transform(
        sentences,
        graph,
        manifest,
        snapshot,
        languages=selected,
        profile=profile,
        max_per_language=max_per_language,
        seed=seed,
        ratios=ratios,
        cross_label_duplicates=cross_label_duplicates,
        initial_rejections=parse_rejections,
        link_counts=link_counts,
        link_samples=link_samples,
    )
    audit["total_source_sentence_rows"] = total_source_rows
    audit["skipped_other_language_rows"] = skipped_other_languages
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "canonical": output_dir / "tatoeba.jsonl",
        "provenance": output_dir / "tatoeba.provenance.jsonl",
        "audit_json": output_dir / "tatoeba.audit.json",
        "audit_text": output_dir / "tatoeba.audit.txt",
    }
    audit["policy"] = decision.to_dict()
    audit["graph_edges_including_reciprocals"] = graph.edge_count
    audit["outputs"] = {name: str(path) for name, path in outputs.items()}
    with tempfile.TemporaryDirectory(
        prefix=".tatoeba-build-", dir=output_dir
    ) as temporary:
        staged = {name: Path(temporary) / path.name for name, path in outputs.items()}
        with staged["canonical"].open("w", encoding="utf-8", newline="\n") as target:
            for record in records:
                target.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        save_sidecars(sidecars, staged["provenance"])
        if validate_jsonl(staged["canonical"]).record_count != len(records):
            raise ValueError("Tatoeba canonical record count mismatch")
        audit["canonical_sha256"] = _sha256(staged["canonical"])
        audit["provenance_sha256"] = _sha256(staged["provenance"])
        staged["audit_json"].write_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        human_lines = [
            f"Tatoeba snapshot {snapshot['snapshot_id']}",
            f"Accepted {len(records)}; rejected {audit['rejected_records']}; component leakage {audit['component_leakage_count']}",
            "The upstream exports are rolling; retain all three archives and their receipt.",
            "",
        ]
        human_lines.extend(
            f"{key.replace('_', ' ').title()}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for key, value in sorted(audit.items())
        )
        staged["audit_text"].write_text("\n".join(human_lines) + "\n", encoding="utf-8")
        for name, destination in outputs.items():
            os.replace(staged[name], destination)
    return audit

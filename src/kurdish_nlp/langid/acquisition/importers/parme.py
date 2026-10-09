"""Pinned, offline-testable PARME Southern Kurdish acquisition and import.

Only :func:`acquire` uses the network, and only when called explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.request import Request, urlopen

from kurdish_nlp.langid.acquisition.licenses import LicenseId
from kurdish_nlp.langid.acquisition.manifests import SourceManifest
from kurdish_nlp.langid.acquisition.policy import PolicyDecision, evaluate
from kurdish_nlp.langid.acquisition.provenance import (
    ProvenanceRecord,
    Transformation,
    save_sidecars,
    validate_linkage,
)
from kurdish_nlp.langid.dataset import DatasetRecord, analyze_records, validate_jsonl
from kurdish_nlp.langid.normalization import normalize_text

SOURCE_FILES = (
    "datasets/SDH-train.tsv",
    "datasets/SDH-val.tsv",
    "datasets/SDH-test.tsv",
)
COLUMNS = (
    "en_sentence",
    "fa_sentence",
    "translation",
    "variety",
    "county",
    "orthography",
    "translator",
)
SOUTHERN_VARIETIES = frozenset({"Pehley", "Kirmashani", "Kalhori", "Garusi", "Badrei"})
ORIGINAL_LABEL = "Southern Kurdish"
ARCHIVE_URL = "https://codeload.github.com/DOLMA-NLP/PARME/tar.gz/{commit}"
DEFAULT_MANIFEST = Path("configs/langid/sources/parme.v1.json")
DEFAULT_RAW_DIR = Path("data/langid/raw/parme")
DEFAULT_OUTPUT_DIR = Path("data/langid/processed")


@dataclass(frozen=True, slots=True)
class RawRow:
    source_file: str
    source_row_number: int
    original_split: str
    english: str
    persian: str
    text: str
    variety: str
    county: str | None
    orthography: str
    translator: str | None
    row_hash: str


@dataclass(frozen=True, slots=True)
class ParseResult:
    rows: tuple[RawRow, ...]
    input_count: int
    rejected: tuple[dict[str, Any], ...]
    rejection_counts: dict[str, int]
    unusual_varieties: dict[str, int]
    invalid_language_labels: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_hash(fields: list[str]) -> str:
    return _text_hash(json.dumps(fields, ensure_ascii=False, separators=(",", ":")))


def _check_manifest(manifest: SourceManifest) -> None:
    if manifest.source_id != "parme":
        raise ValueError("PARME importer requires source_id=parme")
    if manifest.license.identifier is not LicenseId.MIT:
        raise ValueError("PARME importer requires the reviewed MIT license")
    if not manifest.source_url.startswith("https://github.com/DOLMA-NLP/PARME/"):
        raise ValueError(
            "PARME source_url does not match the reviewed upstream repository"
        )
    if len(manifest.source_version) != 40 or any(
        char not in "0123456789abcdef" for char in manifest.source_version
    ):
        raise ValueError("PARME source_version must be a full lowercase Git commit SHA")
    if manifest.checksum_sha256 is None:
        raise ValueError("PARME manifest requires the pinned archive SHA-256")


def archive_path(manifest: SourceManifest, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    _check_manifest(manifest)
    return raw_dir / f"parme-{manifest.source_version}.tar.gz"


def plan_acquisition(
    manifest: SourceManifest,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
) -> dict[str, Any]:
    """Return a network-free plan, including the training policy decision."""
    _check_manifest(manifest)
    decision = evaluate(manifest, role="training", profile=profile)
    return {
        "source_id": manifest.source_id,
        "source_url": manifest.source_url,
        "archive_url": ARCHIVE_URL.format(commit=manifest.source_version),
        "source_version": manifest.source_version,
        "license": manifest.license.identifier.value,
        "destination": str(archive_path(manifest, raw_dir)),
        "expected_sha256": manifest.checksum_sha256,
        "stages": [
            "download",
            "verify_sha256",
            "parse_sdh_splits",
            "normalize",
            "deduplicate",
            "align_groups",
            "split",
            "export",
            "validate",
        ],
        "policy": decision.to_dict(),
    }


def acquire(
    manifest: SourceManifest,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
    opener: Callable[..., Any] | None = None,
) -> tuple[Path, bool]:
    """Fetch the pinned archive; return (path, reused_existing_artifact)."""
    _check_manifest(manifest)
    decision = evaluate(manifest, role="training", profile=profile)
    if not decision.allowed:
        raise ValueError(f"PARME training is blocked: {decision.to_dict()['reasons']}")
    destination = archive_path(manifest, raw_dir)
    expected = manifest.checksum_sha256
    assert expected is not None
    if destination.exists():
        actual = _sha256(destination)
        if actual != expected:
            raise ValueError(
                f"existing PARME archive checksum mismatch: {actual} != {expected}"
            )
        return destination, True

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "kurdish-nlp-parme-importer/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(
        ARCHIVE_URL.format(commit=manifest.source_version), headers=headers
    )
    open_url = opener or urlopen
    with open_url(request, timeout=60) as response:
        status = response.status
        if status not in (200, 206):
            raise ValueError(f"unexpected archive HTTP status: {status}")
        if status == 206 and not offset:
            raise ValueError("server sent a partial archive without a range request")
        mode = "ab" if status == 206 else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output)

    actual = _sha256(partial)
    if actual != expected:
        raise ValueError(
            f"downloaded PARME archive checksum mismatch: {actual} != {expected}"
        )
    os.replace(partial, destination)
    return destination, False


def verify_archive(manifest: SourceManifest, archive: Path) -> str:
    _check_manifest(manifest)
    if not archive.is_file():
        raise FileNotFoundError(
            f"PARME archive not found: {archive}; run acquire parme"
        )
    actual = _sha256(archive)
    if actual != manifest.checksum_sha256:
        raise ValueError(
            f"PARME archive checksum mismatch: {actual} != {manifest.checksum_sha256}"
        )
    return actual


def _read_member(archive: tarfile.TarFile, source_file: str) -> io.TextIOWrapper:
    matches = [
        member
        for member in archive.getmembers()
        if member.name.endswith("/" + source_file) and member.isfile()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {source_file} in PARME archive")
    binary = archive.extractfile(matches[0])
    if binary is None:
        raise ValueError(f"could not read {source_file} from PARME archive")
    return io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")


def parse_archive(archive_path: Path) -> ParseResult:
    """Parse only the three SDH TSV splits. No network or filesystem extraction."""
    rows: list[RawRow] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    unusual_varieties: Counter[str] = Counter()
    invalid_language_labels: Counter[str] = Counter()
    input_count = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for source_file in SOURCE_FILES:
                with _read_member(archive, source_file) as source:
                    reader = csv.reader(source, delimiter="\t", strict=True)
                    header = next(reader, None)
                    if header is None or tuple(header) != COLUMNS:
                        raise ValueError(
                            f"unexpected PARME TSV header in {source_file}: {header!r}"
                        )
                    original_split = source_file.removeprefix(
                        "datasets/SDH-"
                    ).removesuffix(".tsv")
                    for values in reader:
                        input_count += 1
                        line = reader.line_num
                        reason: str | None = None
                        if len(values) == 8 and values[-1] == "":
                            values = values[:7]
                        if len(values) != 7:
                            reason = "malformed_columns"
                        else:
                            (
                                english,
                                persian,
                                text,
                                variety,
                                county,
                                orthography,
                                translator,
                            ) = values
                            if not normalize_text(text):
                                reason = "empty_text"
                            elif not normalize_text(english) or not normalize_text(
                                persian
                            ):
                                reason = "missing_alignment_text"
                            elif orthography.strip() != ORIGINAL_LABEL:
                                reason = "invalid_language_label"
                                invalid_language_labels[
                                    orthography.strip() or "<missing>"
                                ] += 1
                            elif variety.strip() not in SOUTHERN_VARIETIES:
                                reason = "unexpected_variety"
                                unusual_varieties[variety.strip() or "<missing>"] += 1
                        if reason:
                            reasons[reason] += 1
                            if len(rejected) < 25:
                                rejected.append(
                                    {
                                        "source_file": source_file,
                                        "line": line,
                                        "reason": reason,
                                    }
                                )
                            continue
                        rows.append(
                            RawRow(
                                source_file=source_file,
                                source_row_number=line,
                                original_split=original_split,
                                english=english,
                                persian=persian,
                                text=text,
                                variety=variety.strip(),
                                county=county.strip() or None,
                                orthography=orthography.strip(),
                                translator=translator.strip() or None,
                                row_hash=_row_hash(values),
                            )
                        )
    except (tarfile.TarError, csv.Error, UnicodeError) as error:
        raise ValueError(f"invalid PARME archive or TSV: {error}") from error
    return ParseResult(
        tuple(rows),
        input_count,
        tuple(rejected),
        dict(sorted(reasons.items())),
        dict(sorted(unusual_varieties.items())),
        dict(sorted(invalid_language_labels.items())),
    )


def _deduplicate(
    parsed: ParseResult,
) -> tuple[list[RawRow], Counter[str], list[dict[str, Any]]]:
    """Keep deterministic representatives; report every duplicate class."""
    seen_rows: set[str] = set()
    seen_texts: set[str] = set()
    accepted: list[RawRow] = []
    counts: Counter[str] = Counter(parsed.rejection_counts)
    examples = list(parsed.rejected)
    for row in sorted(
        parsed.rows,
        key=lambda item: (item.row_hash, item.source_file, item.source_row_number),
    ):
        reason: str | None = None
        if row.row_hash in seen_rows:
            reason = "duplicate_source_row"
        seen_rows.add(row.row_hash)
        normalized = normalize_text(row.text)
        if reason is None and normalized in seen_texts:
            reason = "duplicate_text"
        if reason:
            counts[reason] += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "source_file": row.source_file,
                        "line": row.source_row_number,
                        "reason": reason,
                    }
                )
            continue
        seen_texts.add(normalized)
        accepted.append(row)
    return accepted, counts, examples


def _alignment_groups(rows: list[RawRow]) -> dict[str, str]:
    """Connect rows sharing either normalized English or Persian source text."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in rows:
        union(
            "en:" + _text_hash(normalize_text(row.english)),
            "fa:" + _text_hash(normalize_text(row.persian)),
        )
    groups: dict[str, str] = {}
    for row in rows:
        root = find("en:" + _text_hash(normalize_text(row.english)))
        groups[row.row_hash] = "parme:alignment:" + _text_hash(root)
    return groups


def _split_for_group(
    group_id: str, seed: int, ratios: tuple[float, float, float]
) -> str:
    score = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    if score < ratios[0]:
        return "train"
    if score < ratios[0] + ratios[1]:
        return "dev"
    return "test"


def _check_ratios(ratios: tuple[float, float, float]) -> None:
    if (
        len(ratios) != 3
        or any(not 0 < value < 1 for value in ratios)
        or abs(sum(ratios) - 1) > 1e-9
    ):
        raise ValueError(
            "train/dev/test ratios must each be between 0 and 1 and sum to 1"
        )


def transform_rows(
    parsed: ParseResult,
    manifest: SourceManifest,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[DatasetRecord], list[ProvenanceRecord], dict[str, Any]]:
    _check_ratios(ratios)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rows, rejections, rejection_examples = _deduplicate(parsed)
    if not rows:
        raise ValueError("PARME import has no usable Southern Kurdish records")
    # Include rejected duplicate-text rows as alignment bridges. Their source
    # sentences can connect otherwise distinct translations across splits.
    groups = _alignment_groups(list(parsed.rows))
    records: list[DatasetRecord] = []
    sidecars: list[ProvenanceRecord] = []
    normalization_changes = 0
    short_count = 0
    long_count = 0
    missing: Counter[str] = Counter()
    original_splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        normalized = normalize_text(row.text)
        normalization_changes += normalized != row.text
        token_count = len(normalized.split())
        short_count += token_count < 3
        long_count += token_count > 100
        if row.county is None:
            missing["county"] += 1
        if row.translator is None:
            missing["translator"] += 1
        group_id = groups[row.row_hash]
        original_splits_by_group[group_id].add(row.original_split)
        canonical_id = f"parme:sdh:{row.row_hash}"
        split = _split_for_group(group_id, seed, ratios)
        records.append(
            DatasetRecord.from_mapping(
                {
                    "id": canonical_id,
                    "text": normalized,
                    "label": "sdh",
                    "source": f"parme:{manifest.source_version}",
                    "domain": "translated_prompt",
                    "document_id": group_id,
                    "license": "MIT",
                    "split": split,
                }
            )
        )
        sidecars.append(
            ProvenanceRecord(
                schema_version=1,
                canonical_record_id=canonical_id,
                source_id="parme",
                acquisition_manifest_version=manifest.manifest_version,
                alignment_group_id=group_id,
                source_file=row.source_file,
                source_row_number=row.source_row_number,
                original_url=f"https://github.com/DOLMA-NLP/PARME/blob/{manifest.source_version}/{row.source_file}#L{row.source_row_number}",
                translator_id=row.translator,
                language_variety=row.variety,
                county=row.county,
                orthography=row.orthography,
                original_language_label=row.orthography,
                original_split=row.original_split,
                original_license="MIT",
                original_text_sha256=_text_hash(row.text),
                transformations=(
                    Transformation(
                        "normalize",
                        "1",
                        parameters={"method": "conservative_nfc_whitespace"},
                    ),
                    Transformation(
                        "group_split",
                        "1",
                        parameters={
                            "seed": seed,
                            "ratios": list(ratios),
                            "alignment": "connected_en_fa_source_text",
                        },
                    ),
                ),
            )
        )
    paired = sorted(zip(records, sidecars, strict=True), key=lambda pair: pair[0].id)
    records = [pair[0] for pair in paired]
    sidecars = [pair[1] for pair in paired]
    validate_linkage(records, sidecars)
    analysis = analyze_records(records)
    if analysis.issues:
        raise ValueError(
            f"canonical PARME records failed validation: {analysis.issues}"
        )

    token_lengths = [len(record.text.split()) for record in records]
    translator_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record, sidecar in zip(records, sidecars, strict=True):
        translator_by_split[record.split][sidecar.translator_id or "<missing>"] += 1
    audit = {
        "source_id": "parme",
        "source_version": manifest.source_version,
        "input_records": parsed.input_count,
        "accepted_records": len(records),
        "rejected_records": parsed.input_count - len(records),
        "duplicate_row_fingerprint_count": rejections["duplicate_source_row"],
        "exact_text_duplicate_count": rejections["duplicate_text"],
        "rejection_counts": dict(sorted(rejections.items())),
        "rejection_examples": rejection_examples,
        "unusual_varieties": parsed.unusual_varieties,
        "invalid_language_labels": parsed.invalid_language_labels,
        "variety_counts": dict(
            sorted(Counter(item.language_variety for item in sidecars).items())
        ),
        "translator_counts": dict(
            sorted(
                Counter(item.translator_id or "<missing>" for item in sidecars).items()
            )
        ),
        "translator_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(translator_by_split.items())
        },
        "generated_split_counts": dict(
            sorted(Counter(item.split for item in records).items())
        ),
        "original_split_counts": dict(
            sorted(Counter(item.original_split for item in sidecars).items())
        ),
        "alignment_group_count": len(original_splits_by_group),
        "groups_crossing_original_splits": sum(
            len(splits) > 1 for splits in original_splits_by_group.values()
        ),
        "normalization_change_count": normalization_changes,
        "very_short_rows_lt_3_tokens": short_count,
        "very_long_rows_gt_100_tokens": long_count,
        "missing_metadata_counts": dict(sorted(missing.items())),
        "text_length_tokens": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "median": median(token_lengths),
        },
        "seed": seed,
        "split_ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
    }
    return records, sidecars, audit


def _write_canonical(records: list[DatasetRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.writelines(
            json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )


def _human_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"PARME Southern Kurdish import ({audit['source_version']})",
        f"Input: {audit['input_records']}  Accepted: {audit['accepted_records']}  Rejected: {audit['rejected_records']}",
        f"Duplicates: source row {audit['duplicate_row_fingerprint_count']}, exact text {audit['exact_text_duplicate_count']}",
        f"Rejection reasons: {audit['rejection_counts']}",
        f"Unexpected varieties: {audit['unusual_varieties']}",
        f"Invalid source labels: {audit['invalid_language_labels']}",
        f"Alignment groups: {audit['alignment_group_count']}  Crossed upstream splits: {audit['groups_crossing_original_splits']}",
        f"Generated splits: {audit['generated_split_counts']}",
        f"Original splits: {audit['original_split_counts']}",
        f"Varieties: {audit['variety_counts']}",
        f"Translators: {audit['translator_counts']}",
        f"Translator distribution by split: {audit['translator_by_split']}",
        f"Token lengths: {audit['text_length_tokens']}",
        f"Short (<3 tokens): {audit['very_short_rows_lt_3_tokens']}  Long (>100 tokens): {audit['very_long_rows_gt_100_tokens']}",
        f"Normalization changes: {audit['normalization_change_count']}",
        f"Missing metadata: {audit['missing_metadata_counts']}",
        f"Source SHA-256: {audit['source_sha256']}",
        f"Canonical SHA-256: {audit['canonical_sha256']}",
        f"Provenance SHA-256: {audit['provenance_sha256']}",
        f"Policy: {audit['policy']}",
    ]
    return "\n".join(lines) + "\n"


def build(
    manifest: SourceManifest,
    archive: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile: str = "commercial",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Verify and import a local archive, then validate the canonical output."""
    decision: PolicyDecision = evaluate(manifest, role="training", profile=profile)
    if not decision.allowed:
        raise ValueError(
            f"PARME import blocked by policy: {decision.to_dict()['reasons']}"
        )
    source_sha = verify_archive(manifest, archive)
    parsed = parse_archive(archive)
    records, sidecars, audit = transform_rows(
        parsed, manifest, seed=seed, ratios=ratios
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "parme.sdh.jsonl"
    provenance_path = output_dir / "parme.provenance.jsonl"
    report_path = output_dir / "parme.audit.json"
    human_path = output_dir / "parme.audit.txt"
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix=".parme-build-", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        staged_canonical = temporary_dir / canonical_path.name
        staged_provenance = temporary_dir / provenance_path.name
        staged_report = temporary_dir / report_path.name
        staged_human = temporary_dir / human_path.name
        _write_canonical(records, staged_canonical)
        save_sidecars(sidecars, staged_provenance)
        validation = validate_jsonl(staged_canonical)
        if validation.record_count != len(records):
            raise ValueError("PARME canonical validation count mismatch")
        audit.update(
            {
                "source_sha256": source_sha,
                "canonical_sha256": _sha256(staged_canonical),
                "provenance_sha256": _sha256(staged_provenance),
                "policy": decision.to_dict(),
                "outputs": {
                    "canonical": str(canonical_path),
                    "provenance": str(provenance_path),
                    "audit_json": str(report_path),
                    "audit_text": str(human_path),
                },
            }
        )
        staged_report.write_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        staged_human.write_text(_human_audit(audit), encoding="utf-8")
        for staged, destination in (
            (staged_canonical, canonical_path),
            (staged_provenance, provenance_path),
            (staged_report, report_path),
            (staged_human, human_path),
        ):
            os.replace(staged, destination)
    return audit

"""Generic, policy-gated Wikimedia encyclopedia importer.

Acquisition is explicit. The build path reads only locally cached, receipted
page revisions and never contacts Wikimedia.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.acquisition.importers.tatoeba import split_for_group
from kurdish_nlp.langid.acquisition.importers.wikimedia_acquire import (
    CONTACT_DEFAULT,
    PAGE_CACHE_NAME,
    Client,
    acquire_api,
    acquire_dump,
    disk_preflight,
    sha256_file,
    verify_receipt,
)
from kurdish_nlp.langid.acquisition.importers.wikimedia_extract import (
    EXTRACTION_VERSION,
    Segment,
    extract_segments,
)
from kurdish_nlp.langid.acquisition.licenses import LicenseId
from kurdish_nlp.langid.acquisition.manifests import SourceManifest, load_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import (
    ProvenanceRecord,
    Transformation,
    validate_parallel_jsonl_linkage,
)
from kurdish_nlp.langid.dataset import DatasetRecord, validate_jsonl
from kurdish_nlp.langid.normalization import normalize_text
from kurdish_nlp.langid.schemas import LanguageCode

DEFAULT_RAW_DIR = Path("data/langid/raw/wikimedia")
DEFAULT_OUTPUT_DIR = Path("data/langid/processed")
MANIFEST_DIR = Path("configs/langid/sources")
PROJECTS = {
    "ckb": ("ckbwiki", "dump"),
    "kmr": ("kuwiki", "dump"),
    "ar": ("arwiki", "api"),
    "fa": ("fawiki", "api"),
    "tr": ("trwiki", "api"),
    "en": ("enwiki", "api"),
}
DUMP_ESTIMATED_BYTES = {"ckb": 87_815_656, "kmr": 49_037_805}


def selected_languages(languages: Iterable[str] | None) -> tuple[str, ...]:
    chosen = tuple(dict.fromkeys(languages or PROJECTS))
    if not chosen or set(chosen) - PROJECTS.keys():
        raise ValueError(
            f"unsupported Wikimedia languages: {sorted(set(chosen) - PROJECTS.keys())}"
        )
    return tuple(sorted(chosen))


def manifests_for(languages: Iterable[str]) -> dict[str, SourceManifest]:
    return {
        label: load_manifest(MANIFEST_DIR / f"wikimedia-{label}.v1.json")
        for label in selected_languages(languages)
    }


def _check_manifest(manifest: SourceManifest, label: str) -> None:
    project, method = PROJECTS[label]
    if manifest.source_id != f"wikimedia-{label}":
        raise ValueError("Wikimedia manifest source ID mismatch")
    expected_url = (
        f"https://dumps.wikimedia.org/{project}/{manifest.source_version}/"
        if method == "dump"
        else f"https://{project.removesuffix('wiki')}.wikipedia.org/w/api.php"
    )
    if manifest.source_url != expected_url:
        raise ValueError("Wikimedia manifest must use the official project URL")
    if method == "dump" and (
        len(manifest.source_version) != 8 or not manifest.source_version.isdigit()
    ):
        raise ValueError("Wikimedia dump version must be an eight-digit date")
    if manifest.license.identifier is not LicenseId.CC_BY_SA_4:
        raise ValueError("Wikimedia importer requires reviewed CC-BY-SA-4.0")


def plan_acquisition(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    max_pages: int = 100,
    profile: str = "commercial",
) -> dict[str, Any]:
    selected = selected_languages(languages)
    upper = 500 if any(PROJECTS[label][1] == "api" for label in selected) else 10_000
    if not 1 <= max_pages <= upper:
        raise ValueError(f"max_pages must be between 1 and {upper} for this selection")
    manifests = manifests_for(selected)
    decisions = {}
    for label, manifest in manifests.items():
        _check_manifest(manifest, label)
        decisions[label] = evaluate(
            manifest, role="training", profile=profile
        ).to_dict()
    estimated = sum(
        DUMP_ESTIMATED_BYTES.get(label, max_pages * 30_000) for label in selected
    )
    return {
        "languages": selected,
        "sources": {
            label: {
                "project": PROJECTS[label][0],
                "method": PROJECTS[label][1],
                "version": manifests[label].source_version,
                "url": manifests[label].source_url,
                "policy": decisions[label],
            }
            for label in selected
        },
        "max_pages_per_language": max_pages,
        "api_requests_estimate_upper_bound": sum(
            1 + (max_pages + 4) // 5
            for label in selected
            if PROJECTS[label][1] == "api"
        ),
        "disk": disk_preflight(raw_dir, estimated),
        "note": "API samples freeze returned page and revision IDs; API acquisition is not an atomic Wikimedia snapshot.",
    }


def acquire(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    max_pages: int = 100,
    seed: int = 42,
    contact: str = CONTACT_DEFAULT,
    profile: str = "commercial",
    interval: float = 1.1,
) -> dict[str, dict[str, Any]]:
    plan = plan_acquisition(
        languages, raw_dir=raw_dir, max_pages=max_pages, profile=profile
    )
    if any(not item["policy"]["allowed"] for item in plan["sources"].values()):
        raise ValueError("Wikimedia acquisition blocked by commercial source policy")
    client = Client(contact, interval=interval)
    receipts: dict[str, dict[str, Any]] = {}
    for label in plan["languages"]:
        manifest = manifests_for((label,))[label]
        project, method = PROJECTS[label]
        if method == "dump":
            receipt = acquire_dump(
                manifest,
                project,
                raw_dir,
                max_pages=max_pages,
                seed=seed,
                client=client,
            )
        else:
            receipt = acquire_api(
                manifest,
                project,
                raw_dir,
                max_pages=max_pages,
                seed=seed,
                client=client,
            )
        receipts[label] = receipt
    return receipts


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    project: str
    page_id: str
    revision_id: str
    revision_timestamp: str
    title: str
    contributor: str | None
    namespace: str
    segment: Segment
    snapshot_id: str
    source_checksum: str
    method: str
    manifest_version: str

    @property
    def document_id(self) -> str:
        return f"wikimedia:{self.project}:page:{self.page_id}"

    @property
    def record_id(self) -> str:
        return f"wikimedia:{self.project}:{self.page_id}:{self.revision_id}:{self.segment.index:04d}"


def _lengths(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": statistics.mean(values) if values else 0,
        "median": statistics.median(values) if values else 0,
    }


def _candidate_order(item: Candidate, seed: int) -> tuple[str, int, str, int]:
    score = int.from_bytes(
        hashlib.sha256(f"{seed}:{item.document_id}".encode()).digest()[:8], "big"
    )
    return item.label, score, item.page_id, item.segment.index


def _cross_source(records: list[DatasetRecord], base: Path) -> dict[str, Any]:
    by_text = {record.text: record for record in records}
    output: dict[str, Any] = {}
    for name, filename in (
        ("tatoeba", "tatoeba.jsonl"),
        ("parme", "parme.sdh.jsonl"),
    ):
        path = base / filename
        if not path.exists():
            output[name] = {"status": "missing"}
            continue
        counts: Counter[str] = Counter()
        examples: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                match = by_text.get(normalize_text(item["text"]))
                if match is None:
                    continue
                counts["normalized_overlap"] += 1
                counts["exact_text_overlap"] += item["text"] == match.text
                counts["label_conflicts"] += item["label"] != match.label.value
                counts["split_conflicts"] += item["split"] != match.split
                if len(examples) < 25:
                    examples.append(
                        {
                            "wikimedia_id": match.id,
                            "other_id": item["id"],
                            "wikimedia_label": match.label.value,
                            "other_label": item["label"],
                            "wikimedia_split": match.split,
                            "other_split": item["split"],
                        }
                    )
        output[name] = {
            "status": "audited",
            "normalized_overlap": counts["normalized_overlap"],
            "exact_text_overlap": counts["exact_text_overlap"],
            "label_conflicts": counts["label_conflicts"],
            "split_conflicts": counts["split_conflicts"],
            "examples": examples,
        }
    return output


def build(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile: str = "commercial",
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    segment_mode: str = "bounded",
    max_tokens: int = 80,
    max_segments_per_page: int = 30,
    max_segments_per_language: int = 50_000,
) -> dict[str, Any]:
    selected = selected_languages(languages)
    if max_segments_per_language < 1:
        raise ValueError("max_segments_per_language must be positive")
    manifests = manifests_for(selected)
    receipts: dict[str, dict[str, Any]] = {}
    decisions = {}
    candidates: list[Candidate] = []
    page_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_counts: Counter[str] = Counter()
    for label in selected:
        manifest = manifests[label]
        _check_manifest(manifest, label)
        decision = evaluate(manifest, role="training", profile=profile)
        decisions[label] = decision.to_dict()
        if not decision.allowed:
            raise ValueError(
                f"Wikimedia import blocked: {decision.to_dict()['reasons']}"
            )
        project, _ = PROJECTS[label]
        receipt = verify_receipt(manifest, project, raw_dir)
        receipts[label] = receipt
        with (raw_dir / project / PAGE_CACHE_NAME).open(encoding="utf-8") as cache:
            for line in cache:
                page = json.loads(line)
                page_counts[label]["examined"] += 1
                if page["namespace"] != "0":
                    page_counts[label]["rejected_namespace"] += 1
                    continue
                if page["redirect"]:
                    page_counts[label]["rejected_redirect"] += 1
                    continue
                segments, reason = extract_segments(
                    page["wikitext"],
                    label=label,
                    mode=segment_mode,
                    max_tokens=max_tokens,
                    max_segments=max_segments_per_page,
                    rejections=rejection_counts,
                )
                if reason:
                    page_counts[label][f"rejected_{reason}"] += 1
                    continue
                page_counts[label]["accepted"] += 1
                page_counts[label]["segments_extracted"] += len(segments)
                for segment in segments:
                    candidates.append(
                        Candidate(
                            label=label,
                            project=project,
                            page_id=page["page_id"],
                            revision_id=page["revision_id"],
                            revision_timestamp=page["revision_timestamp"],
                            title=page["title"],
                            contributor=page.get("contributor"),
                            namespace=page["namespace"],
                            segment=segment,
                            snapshot_id=receipt["snapshot_id"],
                            source_checksum=receipt["pages_sha256"],
                            method=receipt["method"],
                            manifest_version=manifest.manifest_version,
                        )
                    )
    # The full selected candidate set is considered before either deduplication or
    # segment limits, so cross-label identical strings are never silently retained.
    candidates.sort(key=lambda item: _candidate_order(item, seed))
    by_text: dict[str, list[Candidate]] = defaultdict(list)
    for item in candidates:
        by_text[item.segment.text].append(item)
    kept: list[Candidate] = []
    duplicate_examples: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []

    def reject_candidate(
        item: Candidate, reason: str, kept_id: str | None = None
    ) -> None:
        rejected_candidates.append(
            {
                "candidate_id": item.record_id,
                "document_id": item.document_id,
                "label": item.label,
                "page_id": item.page_id,
                "project": item.project,
                "revision_id": item.revision_id,
                "snapshot_id": item.snapshot_id,
                "normalized_text_sha256": hashlib.sha256(
                    item.segment.text.encode("utf-8")
                ).hexdigest(),
                "reason": reason,
                "kept_id": kept_id,
            }
        )

    for text, group in by_text.items():
        labels = {item.label for item in group}
        if len(labels) > 1:
            rejection_counts["cross_label_identical_text"] += len(group)
            for item in group:
                reject_candidate(item, "cross_label_identical_text")
            if len(duplicate_examples) < 20:
                duplicate_examples.append(
                    {
                        "labels": sorted(labels),
                        "ids": [item.record_id for item in group[:8]],
                        "text": text[:120],
                    }
                )
            continue
        kept.append(group[0])
        if len(group) > 1:
            rejection_counts["same_label_duplicate_text"] += len(group) - 1
            for item in group[1:]:
                reject_candidate(item, "same_label_duplicate_text", group[0].record_id)
            if len(duplicate_examples) < 20:
                duplicate_examples.append(
                    {
                        "labels": sorted(labels),
                        "ids": [item.record_id for item in group[:8]],
                        "kept_id": group[0].record_id,
                        "text": text[:120],
                    }
                )
    kept.sort(key=lambda item: _candidate_order(item, seed))
    limited: list[Candidate] = []
    label_counts: Counter[str] = Counter()
    for item in kept:
        if label_counts[item.label] >= max_segments_per_language:
            rejection_counts["segment_limit"] += 1
            reject_candidate(item, "segment_limit")
            continue
        limited.append(item)
        label_counts[item.label] += 1
    limited.sort(key=lambda item: item.record_id)
    records: list[DatasetRecord] = []
    sidecars: list[ProvenanceRecord] = []
    scripts: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)
    characters: dict[str, list[int]] = defaultdict(list)
    length_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    normalization_changes: Counter[str] = Counter()
    suspicious: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contributors: dict[str, Counter[str]] = defaultdict(Counter)
    for item in limited:
        group = item.document_id
        split = split_for_group(group, seed, ratios)
        page_url = f"https://{item.project.removesuffix('wiki')}.wikipedia.org/?curid={item.page_id}"
        revision_url = f"https://{item.project.removesuffix('wiki')}.wikipedia.org/w/index.php?oldid={item.revision_id}"
        record = DatasetRecord(
            id=item.record_id,
            text=item.segment.text,
            label=LanguageCode(item.label),
            source=f"wikimedia-{item.label}:{item.snapshot_id}",
            domain="encyclopedia",
            document_id=group,
            license="CC-BY-SA-4.0",
            split=split,
        )
        sidecar = ProvenanceRecord(
            schema_version=1,
            canonical_record_id=record.id,
            source_id=f"wikimedia-{item.label}",
            acquisition_manifest_version=item.manifest_version,
            original_record_id=f"{item.page_id}:{item.revision_id}:{item.segment.index}",
            original_document_id=group,
            alignment_group_id=group,
            original_url=page_url,
            source_file=f"{item.project}/{PAGE_CACHE_NAME}",
            page_id=item.page_id,
            revision_id=item.revision_id,
            contributor_id=item.contributor,
            original_language_label=item.project.removesuffix("wiki"),
            original_license="CC-BY-SA-4.0",
            original_text_sha256=hashlib.sha256(
                item.segment.original.encode()
            ).hexdigest(),
            normalized_text_sha256=hashlib.sha256(record.text.encode()).hexdigest(),
            source_snapshot=item.snapshot_id,
            transformations=(
                Transformation(
                    name="extract_wikipedia_prose",
                    version=EXTRACTION_VERSION,
                    parameters={
                        "acquisition_method": item.method,
                        "article_title": item.title,
                        "namespace": item.namespace,
                        "revision_timestamp": item.revision_timestamp,
                        "revision_url": revision_url,
                        "section_reference": item.segment.section,
                        "segment_index": item.segment.index,
                        "source_checksum_sha256": item.source_checksum,
                        "quality_flags": list(item.segment.flags),
                        "segment_mode": segment_mode,
                        "contributor_semantics": "revision_editor_not_all_page_authors",
                    },
                ),
            ),
        )
        records.append(record)
        sidecars.append(sidecar)
        contributors[item.label][item.contributor or "<missing>"] += 1
        lengths[item.label].append(len(record.text.split()))
        characters[item.label].append(len(record.text))
        n = lengths[item.label][-1]
        length_buckets[item.label][
            "1-5" if n <= 5 else "6-20" if n <= 20 else "21-50" if n <= 50 else "51+"
        ] += 1
        normalization_changes[item.label] += item.segment.original != item.segment.text
        for flag in item.segment.flags:
            scripts[item.label][flag] += 1
        if item.segment.flags and len(suspicious[item.label]) < 15:
            suspicious[item.label].append(
                {
                    "id": record.id,
                    "flags": item.segment.flags,
                    "text": record.text[:160],
                }
            )
    if not records:
        raise ValueError("Wikimedia import produced no accepted records")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "canonical": output_dir / "wikimedia.jsonl",
        "provenance": output_dir / "wikimedia.provenance.jsonl",
        "rejections": output_dir / "wikimedia.rejections.jsonl",
        "audit_json": output_dir / "wikimedia.audit.json",
        "audit_text": output_dir / "wikimedia.audit.txt",
    }
    audit: dict[str, Any] = {
        "source_id": "wikimedia",
        "languages": list(selected),
        "receipts": {
            label: {
                "project": PROJECTS[label][0],
                "method": receipts[label]["method"],
                "snapshot_id": receipts[label]["snapshot_id"],
                "pages_sha256": receipts[label]["pages_sha256"],
                "source": receipts[label]["source"],
                "sampling": receipts[label]["sampling"],
            }
            for label in selected
        },
        "policy": decisions,
        "pages": {
            label: dict(sorted(page_counts[label].items())) for label in selected
        },
        "segments_extracted": len(candidates),
        "segments_attempted": len(candidates)
        + rejection_counts["insufficient_lexical_content"]
        + rejection_counts["over_max_tokens"]
        + rejection_counts["residual_markup"],
        "segments_accepted": len(records),
        "segments_rejected": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejected_candidate_provenance_count": len(rejected_candidates),
        "duplicate_examples": duplicate_examples,
        "accepted_by_language": dict(
            sorted(Counter(record.label.value for record in records).items())
        ),
        "accepted_by_split": dict(
            sorted(Counter(record.split for record in records).items())
        ),
        "accepted_by_language_split": {
            label: dict(
                sorted(
                    Counter(
                        record.split
                        for record in records
                        if record.label.value == label
                    ).items()
                )
            )
            for label in selected
        },
        "accepted_by_license": {"CC-BY-SA-4.0": len(records)},
        "document_counts": dict(
            sorted(
                {
                    label: len(
                        {
                            record.document_id
                            for record in records
                            if record.label.value == label
                        }
                    )
                    for label in selected
                }.items()
            )
        ),
        "token_lengths": {label: _lengths(lengths[label]) for label in selected},
        "character_lengths": {label: _lengths(characters[label]) for label in selected},
        "length_buckets": {
            label: dict(sorted(length_buckets[label].items())) for label in selected
        },
        "normalization_changes": dict(sorted(normalization_changes.items())),
        "script_flags": {
            label: dict(sorted(scripts[label].items())) for label in selected
        },
        "suspicious_examples": {label: suspicious[label] for label in selected},
        "top_last_revision_editors": {
            label: contributors[label].most_common(10) for label in selected
        },
        "segment_mode": segment_mode,
        "max_tokens": max_tokens,
        "max_segments_per_page": max_segments_per_page,
        "max_segments_per_language": max_segments_per_language,
        "seed": seed,
        "ratios": list(ratios),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    with tempfile.TemporaryDirectory(
        prefix=".wikimedia-build-", dir=output_dir
    ) as temp:
        staged = {name: Path(temp) / path.name for name, path in outputs.items()}
        with (
            staged["canonical"].open("w", encoding="utf-8", newline="\n") as canonical,
            staged["provenance"].open(
                "w", encoding="utf-8", newline="\n"
            ) as provenance,
        ):
            for record, sidecar in zip(records, sidecars, strict=True):
                canonical.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                provenance.write(
                    json.dumps(sidecar.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        with staged["rejections"].open("w", encoding="utf-8", newline="\n") as rejected:
            for item in sorted(
                rejected_candidates, key=lambda value: value["candidate_id"]
            ):
                rejected.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
        analysis = validate_jsonl(staged["canonical"])
        if analysis.record_count != len(records):
            raise ValueError("Wikimedia canonical count mismatch")
        if validate_parallel_jsonl_linkage(
            staged["canonical"], staged["provenance"]
        ) != len(records):
            raise ValueError("Wikimedia provenance linkage count mismatch")
        audit["canonical_sha256"] = sha256_file(staged["canonical"])
        audit["provenance_sha256"] = sha256_file(staged["provenance"])
        audit["rejections_sha256"] = sha256_file(staged["rejections"])
        audit["document_split_leakage"] = len(analysis.document_split_leakage)
        audit["cross_source"] = _cross_source(records, output_dir)
        staged["audit_json"].write_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            f"Wikimedia import: {len(records)} accepted from {len(candidates)} extracted segments",
            "API caches pin individual revisions; they are not an atomic Wikimedia snapshot.",
            "CC-BY-SA-4.0 attribution and share-alike obligations require review before redistribution.",
            "",
        ]
        lines.extend(
            f"{key.replace('_', ' ').title()}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for key, value in sorted(audit.items())
            if key != "suspicious_examples"
        )
        staged["audit_text"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        for name, path in outputs.items():
            os.replace(staged[name], path)
    return audit

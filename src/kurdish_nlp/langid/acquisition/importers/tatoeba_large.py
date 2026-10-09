"""Disk-backed Tatoeba transform for real multi-million-row exports.

The global translation graph remains complete. SQLite bounds candidate and
deduplication memory; canonical and provenance records are emitted one at a time.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from kurdish_nlp.langid.acquisition.importers import tatoeba as t
from kurdish_nlp.langid.acquisition.manifests import SourceManifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import ProvenanceRecord, Transformation
from kurdish_nlp.langid.dataset import DatasetRecord
from kurdish_nlp.langid.normalization import normalize_text


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY, label TEXT NOT NULL, code TEXT NOT NULL,
            text TEXT NOT NULL, raw_hash TEXT NOT NULL, username TEXT,
            created_at TEXT, modified_at TEXT, license_raw TEXT NOT NULL,
            license_source TEXT NOT NULL, source_line INTEGER NOT NULL,
            group_id TEXT NOT NULL, split TEXT NOT NULL, naive_split TEXT NOT NULL,
            normalization_changed INTEGER NOT NULL
        );
        CREATE TABLE accepted_ids (id INTEGER PRIMARY KEY);
    """)


def _insert_batch(connection: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
    )
    batch.clear()


def _deduplicate(
    connection: sqlite3.Connection,
    rejected: Counter[str],
    rejected_by_language: dict[str, Counter[str]],
    *,
    report_examples: bool,
) -> dict[str, Any]:
    exact = normalized_only = cross = cross_exact = cross_normalized_only = 0
    examples: list[dict[str, Any]] = []
    accepted_batch: list[tuple[int]] = []

    def finish(group: list[tuple[str, int, str, str]]) -> None:
        nonlocal exact, normalized_only, cross, cross_exact, cross_normalized_only
        if not group:
            return
        labels = {row[2] for row in group}
        if len(labels) > 1:
            cross += 1
            by_original: dict[str, set[str]] = defaultdict(set)
            for _, _, label, raw_hash in group:
                by_original[raw_hash].add(label)
                rejected_by_language[label]["cross_label_identical_text"] += 1
            if any(len(item) > 1 for item in by_original.values()):
                cross_exact += 1
            else:
                cross_normalized_only += 1
            rejected["cross_label_identical_text"] += len(group)
            if report_examples and len(examples) < 25:
                examples.append(
                    {
                        "text_sha256": t._hash(group[0][0]),
                        "labels": sorted(labels),
                        "sentence_ids": sorted(row[1] for row in group),
                    }
                )
            return
        winner = min(group, key=lambda row: row[1])
        accepted_batch.append((winner[1],))
        for row in group:
            if row[1] == winner[1]:
                continue
            if row[3] == winner[3]:
                exact += 1
            else:
                normalized_only += 1
            rejected["same_label_duplicate_text"] += 1
            rejected_by_language[row[2]]["same_label_duplicate_text"] += 1
        if len(accepted_batch) >= 10000:
            connection.executemany(
                "INSERT INTO accepted_ids VALUES (?)", accepted_batch
            )
            accepted_batch.clear()

    group: list[tuple[str, int, str, str]] = []
    previous: str | None = None
    for text, sentence_id, label, raw_hash in connection.execute(
        "SELECT text,id,label,raw_hash FROM candidates ORDER BY text,id"
    ):
        if previous is not None and text != previous:
            finish(group)
            group = []
        group.append((text, sentence_id, label, raw_hash))
        previous = text
    finish(group)
    if accepted_batch:
        connection.executemany("INSERT INTO accepted_ids VALUES (?)", accepted_batch)
    connection.commit()
    return {
        "exact_same_label_duplicate_count": exact,
        "normalized_same_label_duplicate_count": normalized_only,
        "cross_label_identical_text_count": cross,
        "cross_label_exact_text_count": cross_exact,
        "cross_label_normalized_only_text_count": cross_normalized_only,
        "cross_label_examples": examples,
    }


def _component_stats(connection: sqlite3.Connection) -> tuple[int, float, float, int]:
    sizes: list[int] = []
    multilingual = 0
    for size, language_count in connection.execute(
        "SELECT COUNT(*),COUNT(DISTINCT label) FROM candidates GROUP BY group_id"
    ):
        sizes.append(size)
        multilingual += language_count > 1
    return len(sizes), sum(sizes) / len(sizes), median(sizes), multilingual


def _leakage_count(connection: sqlite3.Connection, column: str) -> int:
    if column not in {"split", "naive_split"}:
        raise ValueError("invalid leakage column")
    return connection.execute(
        f"SELECT COUNT(*) FROM (SELECT c.group_id FROM candidates c "
        f"JOIN accepted_ids a ON a.id=c.id GROUP BY c.group_id "
        f"HAVING COUNT(DISTINCT c.{column})>1)"
    ).fetchone()[0]


def build_large(
    manifest: SourceManifest,
    snapshot: dict[str, Any],
    *,
    raw_dir: Path,
    output_dir: Path,
    languages: tuple[str, ...],
    profile: str,
    seed: int,
    ratios: tuple[float, float, float],
    cross_label_duplicates: str,
) -> dict[str, Any]:
    """Import a real export with bounded RAM; no network and no graph truncation."""
    t._check_ratios(ratios)
    selected = set(t._languages(languages))
    if cross_label_duplicates not in ("report", "exclude"):
        raise ValueError("cross_label_duplicates must be report or exclude")
    decision = evaluate(manifest, role="training", profile=profile)
    if not decision.allowed:
        raise ValueError(f"Tatoeba import blocked: {decision.to_dict()['reasons']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "canonical": output_dir / "tatoeba.jsonl",
        "provenance": output_dir / "tatoeba.provenance.jsonl",
        "audit_json": output_dir / "tatoeba.audit.json",
        "audit_text": output_dir / "tatoeba.audit.txt",
    }
    with tempfile.TemporaryDirectory(
        prefix=".tatoeba-build-", dir=output_dir
    ) as temporary:
        staged = {name: Path(temporary) / path.name for name, path in outputs.items()}
        connection = sqlite3.connect(Path(temporary) / "candidates.sqlite")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        _schema(connection)
        cc0_ids = t._cc0_ids(t._archive_rows(raw_dir / t.FILES[1], "sentences_CC0.csv"))
        graph = t.TranslationGraph()
        for fields in t._archive_rows(raw_dir / t.FILES[2], "links.csv"):
            if len(fields) != 2:
                raise ValueError("malformed official links export row")
            graph.union(t._positive_id(fields[0]), t._positive_id(fields[1]))
        graph_edges = graph.edge_count
        rejected: Counter[str] = Counter()
        rejected_by_language: dict[str, Counter[str]] = defaultdict(Counter)
        raw_counts: Counter[str] = Counter()
        candidate_license_values: Counter[str] = Counter()
        licensed_counts: Counter[str] = Counter()
        content_free_by_language: Counter[str] = Counter()
        content_free_kinds: Counter[str] = Counter()
        license_cache: dict[str, tuple[Any, str | None]] = {}
        source_rows = skipped_other = 0
        batch: list[tuple[Any, ...]] = []
        for line, fields in enumerate(
            t._archive_rows(raw_dir / t.FILES[0], "sentences_detailed.csv"), 1
        ):
            source_rows += 1
            code = fields[1] if len(fields) >= 2 else ""
            label = t.LANGUAGE_MAP.get(code)
            if label not in selected:
                skipped_other += 1
                continue
            raw_counts[label] += 1
            if len(fields) not in (6, 7):
                rejected["malformed_sentence_columns"] += 1
                rejected_by_language[label]["malformed_sentence_columns"] += 1
                continue
            try:
                sentence_id = t._positive_id(fields[0])
            except ValueError:
                rejected["invalid_sentence_id"] += 1
                rejected_by_language[label]["invalid_sentence_id"] += 1
                continue
            if len(fields) == 7:
                license_raw, license_source = fields[6], "explicit_row_field"
            elif sentence_id in cc0_ids:
                license_raw, license_source = "CC0-1.0", "official_CC0_membership"
            else:
                license_raw, license_source = "CC-BY-2.0-FR", "official_export_default"
            candidate_license_values[license_raw or "<missing>"] += 1
            if license_raw not in license_cache:
                license_cache[license_raw] = t._license_decision(
                    manifest, license_raw, profile
                )
            license_id, reason = license_cache[license_raw]
            if reason:
                rejected[reason] += 1
                rejected_by_language[label][reason] += 1
                continue
            text = normalize_text(fields[2])
            if not text or not t._CONTENT.search(text):
                rejected["content_free"] += 1
                rejected_by_language[label]["content_free"] += 1
                content_free_by_language[label] += 1
                content_free_kinds[
                    "numeric_only"
                    if any(char.isdigit() for char in text)
                    else "punctuation_or_empty"
                ] += 1
                continue
            assert license_id is not None
            licensed_counts[license_id.value] += 1
            group_id = graph.group_id(sentence_id)
            batch.append(
                (
                    sentence_id,
                    label,
                    code,
                    text,
                    t._hash(fields[2]),
                    fields[3] or None,
                    fields[4] or None,
                    fields[5] or None,
                    license_raw,
                    license_source,
                    line,
                    group_id,
                    t.split_for_group(group_id, seed, ratios),
                    t.split_for_group(f"tatoeba:sentence:{sentence_id}", seed, ratios),
                    int(text != fields[2]),
                )
            )
            if len(batch) >= 10000:
                try:
                    _insert_batch(connection, batch)
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "duplicate Tatoeba sentence ID in official export"
                    ) from error
        if batch:
            _insert_batch(connection, batch)
        connection.commit()
        del cc0_ids, graph
        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM candidates"
        ).fetchone()[0]
        if not candidate_count:
            raise ValueError("Tatoeba import has no usable records")
        connection.execute("CREATE INDEX candidates_text ON candidates(text,id)")
        connection.execute("CREATE INDEX candidates_group ON candidates(group_id)")
        connection.commit()
        duplicate_audit = _deduplicate(
            connection,
            rejected,
            rejected_by_language,
            report_examples=cross_label_duplicates == "report",
        )
        component_count, mean_size, median_size, multilingual = _component_stats(
            connection
        )
        component_leakage = _leakage_count(connection, "split")
        naive_leakage = _leakage_count(connection, "naive_split")
        if component_leakage:
            raise ValueError(
                f"Tatoeba translation-component leakage: {component_leakage}"
            )
        accepted_by_language: Counter[str] = Counter()
        accepted_by_license: Counter[str] = Counter()
        accepted_by_split: Counter[str] = Counter()
        by_language_split: dict[str, Counter[str]] = defaultdict(Counter)
        contributors: Counter[str] = Counter()
        contributor_splits: dict[str, Counter[str]] = defaultdict(Counter)
        lengths: dict[str, list[int]] = defaultdict(list)
        short: dict[str, Counter[str]] = defaultdict(Counter)
        normalization_changes: Counter[str] = Counter()
        count = 0
        with (
            staged["canonical"].open("w", encoding="utf-8", newline="\n") as canonical,
            staged["provenance"].open(
                "w", encoding="utf-8", newline="\n"
            ) as provenance,
        ):
            for row in connection.execute(
                "SELECT c.* FROM candidates c JOIN accepted_ids a ON a.id=c.id ORDER BY c.label,c.id"
            ):
                (
                    sentence_id,
                    label,
                    code,
                    text,
                    raw_hash,
                    username,
                    created_at,
                    modified_at,
                    license_raw,
                    license_source,
                    source_line,
                    group_id,
                    split,
                    _,
                    changed,
                ) = row
                license_id = t.LICENSE_ALIASES[license_raw]
                record_id = f"tatoeba:{label}:{sentence_id}"
                record = DatasetRecord.from_mapping(
                    {
                        "id": record_id,
                        "text": text,
                        "label": label,
                        "source": f"tatoeba:{snapshot['snapshot_id']}",
                        "domain": "conversational_translation",
                        "document_id": group_id,
                        "license": license_id.value,
                        "split": split,
                    }
                )
                sidecar = ProvenanceRecord(
                    schema_version=1,
                    canonical_record_id=record_id,
                    source_id="tatoeba",
                    acquisition_manifest_version=manifest.manifest_version,
                    original_record_id=str(sentence_id),
                    alignment_group_id=group_id,
                    original_url=f"https://tatoeba.org/en/sentences/show/{sentence_id}",
                    source_file=t.FILES[0],
                    source_row_number=source_line,
                    contributor_id=username,
                    original_language_label=code,
                    original_license=license_raw,
                    original_text_sha256=raw_hash,
                    normalized_text_sha256=t._hash(text),
                    source_snapshot=snapshot["snapshot_id"],
                    transformations=(
                        Transformation(
                            "normalize",
                            "1",
                            parameters={
                                "method": "conservative_nfc_whitespace",
                                "changed": bool(changed),
                            },
                        ),
                        Transformation(
                            "license_resolve",
                            "1",
                            parameters={
                                "source": license_source,
                                "license": license_id.value,
                                "created_at": created_at,
                                "modified_at": modified_at,
                            },
                        ),
                        Transformation(
                            "translation_component_split",
                            "1",
                            parameters={
                                "seed": seed,
                                "ratios": list(ratios),
                                "graph": "all_official_links",
                            },
                        ),
                    ),
                )
                canonical.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                provenance.write(
                    json.dumps(
                        sidecar.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
                accepted_by_language[label] += 1
                accepted_by_license[license_id.value] += 1
                accepted_by_split[split] += 1
                by_language_split[label][split] += 1
                contributor = username or "<missing>"
                contributors[contributor] += 1
                contributor_splits[contributor][split] += 1
                length = len(text.split())
                lengths[label].append(length)
                short[label]["one_token"] += length == 1
                short[label]["one_to_five_tokens"] += length <= 5
                short[label]["url"] += bool(t._URL.search(text))
                normalization_changes[label] += changed
        if not count:
            raise ValueError("Tatoeba import has no accepted records")
        per_language = {
            label: {
                "raw_candidates": raw_counts[label],
                "accepted": accepted_by_language[label],
                "rejected": raw_counts[label] - accepted_by_language[label],
                "train": by_language_split[label]["train"],
                "dev": by_language_split[label]["dev"],
                "test": by_language_split[label]["test"],
            }
            for label in sorted(selected)
        }
        audit = {
            "source_id": "tatoeba",
            "snapshot_id": snapshot["snapshot_id"],
            "acquired_at": snapshot.get("acquired_at"),
            "source_last_modified_by_file": {
                name: snapshot["files"][name].get("last_modified") for name in t.FILES
            },
            "source_checksums": {
                name: snapshot["files"][name]["sha256"] for name in t.FILES
            },
            "languages": sorted(selected),
            "total_source_sentence_rows": source_rows,
            "skipped_other_language_rows": skipped_other,
            "sentences_examined": sum(raw_counts.values()),
            "raw_candidates_by_language": dict(sorted(raw_counts.items())),
            "per_language": per_language,
            "accepted_records": count,
            "rejected_records": sum(raw_counts.values()) - count,
            "rejected_counts": dict(sorted(rejected.items())),
            "rejected_by_language": {
                label: dict(sorted(rejected_by_language[label].items()))
                for label in sorted(selected)
            },
            "accepted_by_language": dict(sorted(accepted_by_language.items())),
            "accepted_by_license": dict(sorted(accepted_by_license.items())),
            "candidate_license_values": dict(sorted(candidate_license_values.items())),
            "licensed_candidates_by_license": dict(sorted(licensed_counts.items())),
            "accepted_by_split": dict(sorted(accepted_by_split.items())),
            "translation_component_count": component_count,
            "component_size_in_target_languages": {
                "mean": mean_size,
                "median": median_size,
            },
            "multilingual_target_components": multilingual,
            "naive_row_split_leakage_components": naive_leakage,
            "component_leakage_count": component_leakage,
            "contributor_counts": dict(sorted(contributors.items())),
            "contributor_split_counts": {
                name: dict(sorted(splits.items()))
                for name, splits in sorted(contributor_splits.items())
            },
            "contributors_spanning_splits": sorted(
                name for name, splits in contributor_splits.items() if len(splits) > 1
            ),
            "dominant_contributors_over_25_percent": {
                name: n for name, n in contributors.items() if n / count > 0.25
            },
            "token_lengths_by_language": {
                label: {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "median": median(values),
                }
                for label, values in sorted(lengths.items())
            },
            "short_text_counts": {
                label: dict(sorted(values.items()))
                for label, values in sorted(short.items())
            },
            "content_free_rejections_by_language": dict(
                sorted(content_free_by_language.items())
            ),
            "content_free_rejection_kinds": dict(sorted(content_free_kinds.items())),
            "normalization_changes_by_language": dict(
                sorted(normalization_changes.items())
            ),
            "sampling_before_by_language": dict(
                sorted(
                    Counter(
                        dict(
                            connection.execute(
                                "SELECT label,COUNT(*) FROM candidates GROUP BY label"
                            )
                        )
                    ).items()
                )
            ),
            "sampling_after_by_language": dict(
                sorted(
                    Counter(
                        dict(
                            connection.execute(
                                "SELECT label,COUNT(*) FROM candidates GROUP BY label"
                            )
                        )
                    ).items()
                )
            ),
            "max_per_language": None,
            "seed": seed,
            "split_ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
            "cross_label_duplicate_policy": cross_label_duplicates,
            "graph_edges_including_reciprocals": graph_edges,
            "policy": decision.to_dict(),
            "outputs": {name: str(path) for name, path in outputs.items()},
            **duplicate_audit,
        }
        audit["canonical_sha256"] = t._sha256(staged["canonical"])
        audit["provenance_sha256"] = t._sha256(staged["provenance"])
        staged["audit_json"].write_text(
            json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        human = [
            f"Tatoeba snapshot {snapshot['snapshot_id']}",
            f"Accepted {count}; rejected {audit['rejected_records']}; component leakage {component_leakage}",
            "The upstream exports are rolling; retain all three archives and their receipt.",
            "",
        ]
        human.extend(
            f"{key.replace('_', ' ').title()}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for key, value in sorted(audit.items())
        )
        staged["audit_text"].write_text("\n".join(human) + "\n", encoding="utf-8")
        connection.close()
        for name, destination in outputs.items():
            os.replace(staged[name], destination)
        return audit

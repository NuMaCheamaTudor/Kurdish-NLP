"""Immutable data release, source decisions, leakage groups and checksum lock."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import audit, bio_errors, parse_bio, sha256, spans
from .grouping import group_records


def dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def split_for(group: str, policy: dict) -> str:
    number = (
        int(hashlib.sha256(f"{policy['seed']}:{group}".encode()).hexdigest(), 16)
        / 2**256
    )
    if number < policy["split_ratios"]["train"]:
        return "train"
    if number < policy["split_ratios"]["train"] + policy["split_ratios"]["dev"]:
        return "dev"
    return "test"


def build(source: Path, policy_path: Path, out: Path, lock_path: Path) -> dict:
    if out.exists() or lock_path.exists():
        raise FileExistsError(
            "Release directory / lock already exists. Verify or choose a new version; never overwrite a frozen split."
        )
    policy = json.loads(policy_path.read_text())
    if abs(sum(policy["split_ratios"].values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    if policy["near_shingle_width"] != 3:
        raise ValueError("This policy implementation requires trigrams")
    source_hash = sha256(source)
    records = parse_bio(source)
    raw_audit, issues = audit(records, policy["long_segment_tokens"])
    groups, edges = group_records(
        records, policy["near_jaccard"], policy["near_min_tokens"]
    )
    text_groups = defaultdict(list)
    flags = defaultdict(set)
    for issue in issues:
        if issue["kind"] == "record_flags":
            flags[issue["record"]].update(issue["flags"])
    for record in records:
        text_groups[tuple(record["tokens"])].append(record)
    conflicting = {
        r["id"]
        for rs in text_groups.values()
        if len({tuple(r["tags"]) for r in rs}) > 1
        for r in rs
    }
    kept, decisions, rejected = [], [], []
    representatives = {}
    for raw in records:
        record = dict(raw)
        corrected = [policy["tag_mapping"].get(tag, tag) for tag in raw["tags"]]
        changes = [
            {
                "index": i,
                "line": raw["row_lines"][i],
                "old": a,
                "new": b,
                "rule": "explicit_tag_mapping",
            }
            for i, (a, b) in enumerate(zip(raw["tags"], corrected))
            if a != b
        ]
        reasons = []
        if raw["malformed"]:
            reasons.append("malformed_rows")
        if raw["id"] in conflicting:
            reasons.append("contradictory_exact_text_annotations")
        if bio_errors(corrected):
            reasons.append("ambiguous_invalid_bio")
        if not raw["tokens"]:
            reasons.append("empty_parsed_record")
        record["tags"] = corrected
        gid = groups[raw["id"]]
        decision = {
            "id": raw["id"],
            "line_start": raw["line_start"],
            "line_end": raw["line_end"],
            "source_sha256": source_hash,
            "group": gid,
            "corrections": changes,
            "flags": sorted(flags[raw["id"]]),
            "reasons": reasons,
        }
        if reasons:
            decision["status"] = "quarantined"
            rejected.append({"original": raw, "decision": decision})
        elif tuple(record["tokens"]) in representatives:
            representative = representatives[tuple(record["tokens"])]
            decision.update(status="deduplicated", representative=representative["id"])
            representative["source_records"].append(raw["id"])
        else:
            decision.update(status="retained", representative=raw["id"])
            record.update(
                group=gid,
                split=split_for(gid, policy),
                source_records=[raw["id"]],
                flags=decision["flags"],
            )
            representatives[tuple(record["tokens"])] = record
            kept.append(record)
        decisions.append(decision)
    for decision in decisions:
        if decision["status"] != "quarantined":
            decision["split"] = split_for(decision["group"], policy)
    stats = {}
    all_labels = sorted({label for r in kept for _, _, label in spans(r["tags"])})
    for split in ("train", "dev", "test"):
        selected = [r for r in kept if r["split"] == split]
        counts = Counter(label for r in selected for _, _, label in spans(r["tags"]))
        stats[split] = {
            "segments": len(selected),
            "tokens": sum(len(r["tokens"]) for r in selected),
            "groups": len({r["group"] for r in selected}),
            "entities": {label: counts[label] for label in all_labels},
        }
    insufficient = {
        label: {split: stats[split]["entities"][label] for split in stats}
        for label in all_labels
        if stats["train"]["entities"][label] < policy["insufficient_train_entities"]
        or any(
            stats[split]["entities"][label] < policy["insufficient_eval_entities"]
            for split in ("dev", "test")
        )
    }
    report = {
        "version": policy["version"],
        "source_sha256": source_hash,
        "raw": raw_audit,
        "cleaning": {
            "status_counts": dict(Counter(d["status"] for d in decisions)),
            "correction_counts": dict(
                Counter(
                    c["old"] + " -> " + c["new"]
                    for d in decisions
                    for c in d["corrections"]
                )
            ),
            "quarantine_reasons": dict(
                Counter(reason for d in decisions for reason in d["reasons"])
            ),
            "repaired_bio_transitions": 0,
        },
        "grouping": {
            "edge_counts": dict(Counter(e["kind"] for e in edges)),
            "groups_including_quarantine": len(set(groups.values())),
            "note": "Transitive components include rejected records; near similarity is lexical, not document identity. Short records use normalized equality only.",
        },
        "splits": stats,
        "labels": all_labels,
        "insufficient_support": insufficient,
        "limitations": [
            "No verified article, document, author, date or publisher IDs: no document-independent evaluation claim.",
            "Boilerplate candidates are repeated prefixes/suffixes, not confirmed news-source attribution.",
            "Near-duplicate detector does not establish paraphrase or topic independence.",
            "Original whitespace is absent: text reconstructed with one ASCII space between BIO tokens.",
            "Source licensing / annotation guidelines / annotation agreement remain unverified.",
            "Historical 88.17 F1 is contaminated and excluded from V2 comparisons.",
        ],
    }
    # Test annotations are used only for dataset auditing/support, never model selection.
    out.mkdir(parents=True)
    dump(out / "policy.json", policy)
    dump(out / "audit.json", report)
    write_jsonl(out / "audit-issues.jsonl", issues)
    write_jsonl(out / "group-edges.jsonl", edges)
    write_jsonl(out / "provenance.jsonl", decisions)
    write_jsonl(out / "quarantine.jsonl", rejected)
    write_jsonl(out / "cleaned.jsonl", kept)
    for split in stats:
        write_jsonl(out / f"{split}.jsonl", (r for r in kept if r["split"] == split))
    write_jsonl(
        out / "split-manifest.jsonl",
        (
            {
                "id": r["id"],
                "text_sha256": r["text_sha256"],
                "group": r["group"],
                "split": r["split"],
                "source_records": r["source_records"],
            }
            for r in kept
        ),
    )
    if sha256(source) != source_hash:
        raise ValueError("Source changed during build")
    lock = {
        "version": policy["version"],
        "source_sha256": source_hash,
        "policy_sha256": sha256(policy_path),
        "files": {p.name: sha256(p) for p in sorted(out.iterdir())},
        "segments": {s: stats[s]["segments"] for s in stats},
        "labels": all_labels,
        "evaluation_scope": "lexical-duplicate-group-held-out segments; original documents unknown",
    }
    dump(out / "release.json", lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    dump(lock_path, lock)
    return report


def verify(out: Path, lock_path: Path, source: Path | None = None) -> dict:
    lock = json.loads(lock_path.read_text())
    if json.loads((out / "release.json").read_text()) != lock:
        raise ValueError("Release metadata differs from frozen lock")
    for name, digest in lock["files"].items():
        if sha256(out / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    if source is not None and sha256(source) != lock["source_sha256"]:
        raise ValueError("Original source checksum mismatch")
    seen_text, seen_groups, seen_ids, seen_refs = {}, {}, set(), set()
    for split in ("train", "dev", "test"):
        for r in read_jsonl(out / f"{split}.jsonl"):
            if (
                r["id"] in seen_ids
                or bio_errors(r["tags"])
                or len(r["tokens"]) != len(r["tags"])
            ):
                raise ValueError("Duplicate ID or invalid training record")
            seen_ids.add(r["id"])
            if r["split"] != split:
                raise ValueError("Split assignment mismatch")
            if r["text_sha256"] in seen_text:
                raise ValueError("Repeated text in deduplicated release")
            seen_text[r["text_sha256"]] = split
            if r["group"] in seen_groups and seen_groups[r["group"]] != split:
                raise ValueError("Group crosses split boundary")
            seen_groups[r["group"]] = split
            for ref in r["source_records"]:
                if ref in seen_refs:
                    raise ValueError("Source record assigned more than once")
                seen_refs.add(ref)
    return lock

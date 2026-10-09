"""Offline contract tests for the generic Tatoeba acquisition and importer."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from kurdish_nlp.langid.acquisition.importers import tatoeba
from kurdish_nlp.langid.acquisition.manifests import load_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import (
    load_sidecars,
    validate_parallel_jsonl_linkage,
)
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import validate_jsonl

MANIFEST = load_manifest(tatoeba.DEFAULT_MANIFEST)
FIXTURE = json.loads(Path("tests/fixtures/langid/tatoeba/synthetic.json").read_text())


def _snapshot(raw_dir: Path, *, reverse: bool = False) -> dict:
    raw_dir.mkdir(parents=True)
    members = {
        "sentences_detailed.tar.bz2": (
            "sentences_detailed.csv",
            FIXTURE["sentences_detailed"],
        ),
        "sentences_CC0.tar.bz2": ("sentences_CC0.csv", FIXTURE["sentences_CC0"]),
        "links.tar.bz2": ("links.csv", FIXTURE["links"]),
    }
    files = {}
    for archive_name, (member_name, rows) in members.items():
        content = (
            "".join("\t".join(row) + "\n" for row in reversed(rows) if reverse)
            if reverse
            else "".join("\t".join(row) + "\n" for row in rows)
        )
        encoded = content.encode()
        with tarfile.open(raw_dir / archive_name, "w:bz2") as archive:
            info = tarfile.TarInfo(member_name)
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        files[archive_name] = {
            "url": tatoeba.EXPORT_ROOT + archive_name,
            "sha256": hashlib.sha256((raw_dir / archive_name).read_bytes()).hexdigest(),
        }
    snapshot_id = hashlib.sha256(
        json.dumps(
            {name: files[name]["sha256"] for name in tatoeba.FILES},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt = {"schema_version": 1, "snapshot_id": snapshot_id, "files": files}
    (raw_dir / tatoeba.RECEIPT_NAME).write_text(json.dumps(receipt))
    return receipt


def test_manifest_policy_and_dry_run(capsys):
    assert evaluate(MANIFEST, role="training", profile="commercial").allowed
    assert main(["acquire", "tatoeba", "--dry-run"]) == 0
    assert "sentences_CC0.tar.bz2" in capsys.readouterr().out


def test_missing_files_and_checksums(tmp_path):
    with pytest.raises(FileNotFoundError, match="receipt missing"):
        tatoeba.build(MANIFEST, raw_dir=tmp_path)
    _snapshot(tmp_path / "raw")
    (tmp_path / "raw" / "links.tar.bz2").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        tatoeba.build(MANIFEST, raw_dir=tmp_path / "raw")
    missing = tmp_path / "missing"
    _snapshot(missing)
    (missing / "sentences_CC0.tar.bz2").unlink()
    with pytest.raises(FileNotFoundError, match="raw file missing"):
        tatoeba.build(MANIFEST, raw_dir=missing)


def test_graph_transitivity_and_input_order():
    first = tatoeba.TranslationGraph()
    second = tatoeba.TranslationGraph()
    links = [(1, 100), (100, 2), (2, 3), (4, 5)]
    for left, right in links:
        first.union(left, right)
    for left, right in reversed(links):
        second.union(left, right)
    assert first.group_id(3) == second.group_id(3) == "tatoeba:translation-group:1"
    assert first.group_id(4) != first.group_id(3)


def test_row_parsing_and_licensing():
    rejected = Counter()
    rows = tatoeba.parse_sentences(
        FIXTURE["sentences_detailed"], {3}, rejection_counts=rejected
    )
    assert {row.label for row in rows} == tatoeba.TARGETS
    assert rejected["unsupported_language_code"] == 1
    assert next(row for row in rows if row.sentence_id == 3).license_raw == "CC0-1.0"
    assert next(row for row in rows if row.sentence_id == 2).language_code == "pes"
    assert tatoeba._license_decision(MANIFEST, "CC-BY-NC-SA-4.0", "commercial")[1]
    assert tatoeba._license_decision(MANIFEST, "CC-BY-ND-4.0", "commercial")[1]
    assert tatoeba._license_decision(MANIFEST, "CC-BY-NC-ND-4.0", "research")[1]
    assert tatoeba._license_decision(MANIFEST, "", "commercial")[1] == "missing_license"


def test_build_subset_split_and_provenance(tmp_path):
    raw = tmp_path / "raw"
    _snapshot(raw)
    all_output = tmp_path / "all"
    audit = tatoeba.build(MANIFEST, raw_dir=raw, output_dir=all_output)
    assert audit["component_leakage_count"] == 0
    assert (
        "Naive Row Split Leakage Components"
        in (all_output / "tatoeba.audit.txt").read_text()
    )
    assert audit["accepted_by_license"]["CC0-1.0"] == 1
    assert audit["rejected_counts"]["cross_label_identical_text"] == 2
    assert audit["rejected_counts"]["content_free"] == 1
    assert audit["rejected_counts"]["missing_license"] == 1
    assert audit["rejected_counts"]["license_commercial_use_not_allowed"] == 2
    records = validate_jsonl(all_output / "tatoeba.jsonl")
    assert records.record_count == audit["accepted_records"]
    sidecars = load_sidecars(all_output / "tatoeba.provenance.jsonl")
    assert sidecars[0].normalized_text_sha256
    assert sidecars[0].source_snapshot == audit["snapshot_id"]
    assert any(sidecar.contributor_id == "user_a" for sidecar in sidecars)
    rows = [
        json.loads(line)
        for line in (all_output / "tatoeba.jsonl").read_text().splitlines()
    ]
    all_splits = {row["id"]: row["split"] for row in rows}
    assert (
        all_splits["tatoeba:ckb:1"]
        == all_splits["tatoeba:fa:2"]
        == all_splits["tatoeba:en:3"]
    )
    subset = tmp_path / "subset"
    tatoeba.build(MANIFEST, raw_dir=raw, output_dir=subset, languages=("ckb", "fa"))
    subset_rows = [
        json.loads(line) for line in (subset / "tatoeba.jsonl").read_text().splitlines()
    ]
    assert {row["label"] for row in subset_rows} == {"ckb", "fa"}
    assert all_splits["tatoeba:ckb:1"] == next(
        row["split"] for row in subset_rows if row["id"] == "tatoeba:ckb:1"
    )
    assert all_splits["tatoeba:fa:2"] == next(
        row["split"] for row in subset_rows if row["id"] == "tatoeba:fa:2"
    )
    assert "tatoeba:en:15" in all_splits  # one-token linguistic text survives
    assert any(row["text"] == 'He said "hello".' for row in rows)


def test_research_profile_still_rejects_nd(tmp_path):
    raw = tmp_path / "raw"
    _snapshot(raw)
    audit = tatoeba.build(
        MANIFEST, raw_dir=raw, output_dir=tmp_path / "research", profile="research"
    )
    assert audit["accepted_by_license"]["CC-BY-NC-SA-4.0"] == 1
    assert audit["rejected_counts"]["license_derivatives_prohibited"] == 1


def test_determinism_sampling_cli(tmp_path):
    raw = tmp_path / "raw"
    _snapshot(raw)
    left, right = tmp_path / "left", tmp_path / "right"
    assert (
        main(
            [
                "import",
                "tatoeba",
                "--raw-dir",
                str(raw),
                "--output-dir",
                str(left),
                "--languages",
                "ckb",
                "kmr",
                "sdh",
                "ar",
                "fa",
                "tr",
                "en",
                "--max-per-language",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "import",
                "tatoeba",
                "--raw-dir",
                str(raw),
                "--output-dir",
                str(right),
                "--max-per-language",
                "2",
            ]
        )
        == 0
    )
    assert (left / "tatoeba.jsonl").read_bytes() == (
        right / "tatoeba.jsonl"
    ).read_bytes()
    assert (left / "tatoeba.provenance.jsonl").read_bytes() == (
        right / "tatoeba.provenance.jsonl"
    ).read_bytes()
    capped = json.loads((left / "tatoeba.audit.json").read_text())
    assert all(count <= 2 for count in capped["sampling_after_by_language"].values())
    assert main(["import", "tatoeba", "--raw-dir", str(raw), "--languages", "hac"]) == 1


def test_import_split_independent_of_input_order(tmp_path):
    first_raw, second_raw = tmp_path / "first_raw", tmp_path / "second_raw"
    _snapshot(first_raw)
    _snapshot(second_raw, reverse=True)
    first_out, second_out = tmp_path / "first_out", tmp_path / "second_out"
    tatoeba.build(MANIFEST, raw_dir=first_raw, output_dir=first_out)
    tatoeba.build(MANIFEST, raw_dir=second_raw, output_dir=second_out)

    def groups(path):
        return {
            row["id"]: (row["document_id"], row["split"])
            for row in map(json.loads, path.read_text().splitlines())
        }

    assert groups(first_out / "tatoeba.jsonl") == groups(second_out / "tatoeba.jsonl")


def test_disk_backed_path_matches_canonical_fixture(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _snapshot(raw)
    normal, bounded = tmp_path / "normal", tmp_path / "bounded"
    tatoeba.build(MANIFEST, raw_dir=raw, output_dir=normal)
    monkeypatch.setattr(tatoeba, "LARGE_ARCHIVE_THRESHOLD", 0)
    audit = tatoeba.build(MANIFEST, raw_dir=raw, output_dir=bounded)
    assert (normal / "tatoeba.jsonl").read_bytes() == (
        bounded / "tatoeba.jsonl"
    ).read_bytes()
    assert audit["component_leakage_count"] == 0
    assert (
        audit["accepted_records"]
        == validate_jsonl(bounded / "tatoeba.jsonl").record_count
    )
    sidecars = load_sidecars(bounded / "tatoeba.provenance.jsonl")
    assert len(sidecars) == audit["accepted_records"]
    assert (
        validate_parallel_jsonl_linkage(
            bounded / "tatoeba.jsonl", bounded / "tatoeba.provenance.jsonl"
        )
        == audit["accepted_records"]
    )


def test_acquire_mocked_fetch(tmp_path):
    source = tmp_path / "source"
    _snapshot(source)
    destination = tmp_path / "destination"

    class Response(io.BytesIO):
        status = 200

        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.headers: dict[str, str] = {}

    def opener(request, timeout):
        return Response((source / Path(request.full_url).name).read_bytes())

    receipt, reused = tatoeba.acquire(MANIFEST, raw_dir=destination, opener=opener)
    assert not reused
    assert receipt["snapshot_id"]
    assert tatoeba.acquire(MANIFEST, raw_dir=destination, opener=opener)[1]

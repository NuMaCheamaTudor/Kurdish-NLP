"""PARME import tests use only a synthetic local archive and a fake HTTP opener."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from kurdish_nlp.langid.acquisition.importers.parme import (
    COLUMNS,
    SOURCE_FILES,
    acquire,
    archive_path,
    build,
    parse_archive,
    transform_rows,
    verify_archive,
)
from kurdish_nlp.langid.acquisition.manifests import load_manifest, save_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import load_sidecars, validate_linkage
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import load_jsonl, validate_jsonl

REPO = Path(__file__).parents[1]
MANIFEST = REPO / "configs/langid/sources/parme.v1.json"
ROWS = REPO / "tests/fixtures/langid/parme/synthetic_rows.json"


def synthetic_archive(tmp_path: Path) -> Path:
    fixture_rows = json.loads(ROWS.read_text(encoding="utf-8"))
    archive = tmp_path / "parme-fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for source_file in SOURCE_FILES:
            split = source_file.removeprefix("datasets/SDH-").removesuffix(".tsv")
            lines = ["\t".join(COLUMNS)]
            for row in fixture_rows:
                if row["split"] == split:
                    if "malformed_fields" in row:
                        lines.append("\t".join(row["malformed_fields"]))
                    else:
                        lines.append(
                            "\t".join(row[column] for column in COLUMNS) + "\t"
                        )
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            info = tarfile.TarInfo("PARME-fixture/" + source_file)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    return archive


def fixture_manifest(archive: Path):
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    return replace(load_manifest(MANIFEST), checksum_sha256=checksum)


def test_real_manifest_valid_and_commercially_allowed() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.source_version == "6df9269acc75377ed6be3bf2f7966c8e238a62bd"
    assert (
        manifest.checksum_sha256
        == "747ff3f5d23f320a193caebbae0ff01104f20d23ab5891e3810b807a0e19c83d"
    )
    assert evaluate(manifest, role="training", profile="commercial").allowed


def test_parser_filters_and_reports_quality(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    parsed = parse_archive(archive)
    assert parsed.input_count == 18
    assert len(parsed.rows) == 14
    assert parsed.rejection_counts == {
        "empty_text": 1,
        "invalid_language_label": 1,
        "unexpected_variety": 1,
        "malformed_columns": 1,
    }
    assert parsed.invalid_language_labels == {"Hawrami": 1}
    assert parsed.unusual_varieties == {"Hawrami": 1}
    manifest = fixture_manifest(archive)
    records, sidecars, audit = transform_rows(parsed, manifest)
    assert len(records) == 12
    assert {item.label.value for item in records} == {"sdh"}
    assert audit["duplicate_row_fingerprint_count"] == 1
    assert audit["exact_text_duplicate_count"] == 1
    assert audit["normalization_change_count"] == 1
    assert {item.language_variety for item in sidecars} == {
        "Pehley",
        "Kirmashani",
        "Kalhori",
        "Garusi",
        "Badrei",
    }
    assert {item.translator_id for item in sidecars} >= {"T1", "T2", "T3", "T4", "T5"}
    assert any(item.county is None for item in sidecars)
    assert all(item.original_split in {"train", "val", "test"} for item in sidecars)
    assert all(item.orthography == "Southern Kurdish" for item in sidecars)
    assert all(item.source_row_number >= 2 for item in sidecars)


def test_alignment_groups_and_splits_are_deterministic(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    parsed = parse_archive(archive)
    manifest = fixture_manifest(archive)
    first, sidecars, audit = transform_rows(parsed, manifest, seed=19)
    second, second_sidecars, second_audit = transform_rows(parsed, manifest, seed=19)
    assert first == second
    assert sidecars == second_sidecars
    assert audit == second_audit
    id_to_record = {record.id: record for record in first}
    groups: dict[str, set[str]] = {}
    for sidecar in sidecars:
        groups.setdefault(sidecar.alignment_group_id, set()).add(
            id_to_record[sidecar.canonical_record_id].split
        )
    assert all(len(splits) == 1 for splits in groups.values())
    assert audit["groups_crossing_original_splits"] >= 1
    assert len(first) == len({record.id for record in first})
    assert len({record.document_id for record in first}) < len(first)
    validate_linkage(first, sidecars)


def test_build_exports_and_rebuilds_identically(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    manifest = fixture_manifest(archive)
    output = tmp_path / "processed"
    first = build(manifest, archive, output_dir=output)
    canonical = output / "parme.sdh.jsonl"
    provenance = output / "parme.provenance.jsonl"
    assert validate_jsonl(canonical).record_count == 12
    validate_linkage(load_jsonl(canonical), load_sidecars(provenance))
    assert (
        first["canonical_sha256"] == hashlib.sha256(canonical.read_bytes()).hexdigest()
    )
    assert (
        (output / "parme.audit.txt")
        .read_text(encoding="utf-8")
        .startswith("PARME Southern Kurdish import")
    )
    snapshot = {
        item.name: item.read_bytes() for item in output.iterdir() if item.is_file()
    }
    second = build(manifest, archive, output_dir=output)
    assert second == first
    assert snapshot == {
        item.name: item.read_bytes() for item in output.iterdir() if item.is_file()
    }


def test_checksum_mismatch_and_missing_source_fail(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    manifest = fixture_manifest(archive)
    with pytest.raises(FileNotFoundError, match="run acquire"):
        verify_archive(manifest, tmp_path / "missing.tar.gz")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(load_manifest(MANIFEST), archive)


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, status: int) -> None:
        super().__init__(content)
        self.status = status


def test_mock_acquisition_reuse_and_resume(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    content = archive.read_bytes()
    manifest = fixture_manifest(archive)
    raw_dir = tmp_path / "raw"
    calls: list[str | None] = []

    def opener(request, *, timeout: int):
        assert timeout == 60
        calls.append(request.get_header("Range"))
        return FakeResponse(content, 200)

    destination, reused = acquire(manifest, raw_dir=raw_dir, opener=opener)
    assert destination == archive_path(manifest, raw_dir)
    assert not reused
    assert calls == [None]
    reused_path, reused = acquire(
        manifest,
        raw_dir=raw_dir,
        opener=lambda *_args, **_kwargs: pytest.fail("network used"),
    )
    assert reused and reused_path == destination
    destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(content[:100])

    def resume_opener(request, *, timeout: int):
        assert request.get_header("Range") == "bytes=100-"
        return FakeResponse(content[100:], 206)

    resumed, reused = acquire(manifest, raw_dir=raw_dir, opener=resume_opener)
    assert resumed.read_bytes() == content
    assert not reused


def test_mock_acquisition_rejects_wrong_bytes(tmp_path: Path) -> None:
    archive = synthetic_archive(tmp_path)
    manifest = fixture_manifest(archive)
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire(
            manifest,
            raw_dir=tmp_path / "raw",
            opener=lambda *_args, **_kwargs: FakeResponse(b"bad", 200),
        )


def test_cli_dry_run_and_offline_fixture_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import kurdish_nlp.langid.acquisition.importers.parme as parme_module

    monkeypatch.setattr(
        parme_module, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used")
    )
    assert main(["acquire", "parme", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["policy"]["allowed"] is True
    assert plan["source_version"] == load_manifest(MANIFEST).source_version

    archive = synthetic_archive(tmp_path)
    manifest = fixture_manifest(archive)
    manifest_path = tmp_path / "fixture-manifest.json"
    save_manifest(manifest, manifest_path)
    output = tmp_path / "output"
    assert (
        main(
            [
                "import",
                "parme",
                "--manifest",
                str(manifest_path),
                "--archive",
                str(archive),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert "12 accepted" in capsys.readouterr().out
    assert validate_jsonl(output / "parme.sdh.jsonl").record_count == 12

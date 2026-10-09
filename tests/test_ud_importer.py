"""Offline tests for the separate UD external-evaluation path."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from kurdish_nlp.langid.acquisition.benchmark import (
    ExternalBenchmarkRecord,
    validate_benchmark,
)
from kurdish_nlp.langid.acquisition.importers import ud
from kurdish_nlp.langid.acquisition.manifests import load_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import load_jsonl

FIXTURE = Path(__file__).parent / "fixtures/langid/ud/synthetic.conllu"


def test_official_manifests_are_evaluation_only() -> None:
    for language, manifest in ud.manifests_for(("sdh", "kmr")).items():
        ud._check_manifest(manifest, language)
        assert evaluate(
            manifest, role="external_evaluation", profile="commercial"
        ).allowed
        assert evaluate(manifest, role="benchmark", profile="commercial").allowed
        for role in ("training", "development", "calibration", "test"):
            assert not evaluate(manifest, role=role, profile="commercial").allowed


def test_cli_dry_run_is_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["acquire", "ud", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["evaluation_only"] is True
    assert set(plan["sources"]) == {"sdh", "kmr"}
    assert all(item["policy"]["allowed"] for item in plan["sources"].values())


def test_parser_metadata_mwt_empty_node_reconstruction_and_errors() -> None:
    parsed = list(ud.parse_conllu(FIXTURE.open(encoding="utf-8")))
    assert len(parsed) == 5
    first = parsed[0][0]
    assert first is not None
    assert (first.document_id, first.paragraph_id, first.sent_id) == (
        "doc-1",
        "par-1",
        "sample-1",
    )
    assert first.text == "Ezê biçim."
    assert first.multiword_count == 1
    assert first.empty_node_count == 1
    assert first.reconstruction == "official_text"
    second = parsed[1][0]
    assert second is not None and second.text == "ئەم دەچین."
    assert second.reconstruction == "reconstructed"
    assert "missing_or_empty_text" in second.warnings
    assert parsed[2][0] is not None and parsed[2][0].sent_id == "sample-2"
    assert parsed[3][0] is not None and "missing_sent_id" in parsed[3][0].warnings
    assert parsed[4][0] is None and "10 columns" in parsed[4][1]["reason"]
    assert ud._script(second.text)["label"] == "arabic"
    assert ud._script(first.text)["label"] == "latin"


def test_missing_form_inside_multiword_uses_official_text() -> None:
    data = "# text = Jê re.\n1-2\tJê\t_\t_\t_\t_\t_\t_\t_\t_\n1\tJê\t_\tX\t_\t_\t0\troot\t_\t_\n2\t_\t_\tX\t_\t_\t1\tdep\t_\t_\n3\tre\t_\tX\t_\t_\t1\tdep\t_\tSpaceAfter=No\n4\t.\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
    sentence, problem = next(ud.parse_conllu(io.StringIO(data)))
    assert problem is None
    assert sentence is not None and sentence.text == "Jê re."
    assert "missing_token_form_in_annotations" in sentence.warnings


def test_reconstruction_uses_multiword_surface_and_spacing() -> None:
    data = (
        "1-2\tJê\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tJi\t_\tX\t_\t_\t0\troot\t_\t_\n"
        "2\t_\t_\tX\t_\t_\t1\tdep\t_\tSpaceAfter=No\n"
        "3\t,\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
        "4\tre\t_\tX\t_\t_\t1\tdep\t_\tSpaceAfter=No\n"
        "5\t.\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
    )
    sentence, problem = next(ud.parse_conllu(io.StringIO(data)))
    assert problem is None
    assert sentence is not None and sentence.text == "Jê, re."


def test_strict_benchmark_schema_rejects_training_export(tmp_path: Path) -> None:
    record = ExternalBenchmarkRecord(
        "ud:sdh_garrusi:test:1",
        "Latin text",
        "sdh",
        "ud-kurdish-v2.18",
        "sdh_garrusi",
        "test",
        "external_test",
        "universal-dependencies",
        "CC-BY-SA-4.0",
        "sdh_garrusi_original_orthography",
        True,
        (),
    )
    assert ExternalBenchmarkRecord.from_mapping(record.to_dict()) == record
    data = tmp_path / "benchmark.jsonl"
    data.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_jsonl(data)
    bad = record.to_dict()
    bad["evaluation_partition"] = "test"
    with pytest.raises(ValueError, match="external evaluation"):
        ExternalBenchmarkRecord.from_mapping(bad)


def test_provenance_linkage_detects_tampered_text(tmp_path: Path) -> None:
    record = ExternalBenchmarkRecord(
        "ud:sdh_garrusi:test:1",
        "Latin text",
        "sdh",
        "ud-kurdish-v2.18",
        "sdh_garrusi",
        "test",
        "external_test",
        "universal-dependencies",
        "CC-BY-SA-4.0",
        "sdh_garrusi_original_orthography",
        True,
        (),
    )
    data = tmp_path / "data.jsonl"
    sidecar = tmp_path / "provenance.jsonl"
    data.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    sidecar.write_text(
        json.dumps(
            {
                "evaluation_record_id": record.id,
                "treebank": record.treebank,
                "original_split": record.original_split,
                "license": record.license,
                "ud_release": "v2.18",
                "benchmark_text_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text hash mismatch"):
        validate_benchmark(data, sidecar)


def _synthetic_archive(path: Path, language: str) -> str:
    treebank, repository, splits = ud.TREEBANKS[language]
    payload = FIXTURE.read_bytes()
    with tarfile.open(path, "w:gz") as archive:
        for split in splits:
            name = f"{repository}-r2.18/{treebank}-ud-{split}.conllu"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return ud.sha256_file(path)


def test_offline_build_overlap_linkage_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    manifests = {}
    for language in ("sdh", "kmr"):
        digest = _synthetic_archive(ud.archive_path(language, raw), language)
        original = load_manifest(
            ud.MANIFEST_DIR
            / f"ud-{language}-{ud.TREEBANKS[language][0].split('_')[1]}-v2.18.v1.json"
        )
        manifests[language] = replace(original, checksum_sha256=digest)
    monkeypatch.setattr(
        ud,
        "manifests_for",
        lambda languages: {language: manifests[language] for language in languages},
    )
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "parme.sdh.jsonl").write_text(
        json.dumps({"id": "parme:1", "text": "Ezê biçim.", "label": "sdh"}) + "\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "kmr_kurmanji-ud-train.conllu").write_bytes(FIXTURE.read_bytes())
    output = tmp_path / "benchmark"
    first = ud.build(
        ("sdh", "kmr"),
        raw_dir=raw,
        output_dir=output,
        canonical_dir=canonical,
        legacy_dir=legacy,
    )
    assert first["audit"]["record_count"] == 20
    assert first["audit"]["canonical_comparisons_complete"] is False
    assert first["audit"]["strict_external_count"] == 0
    assert all(
        "canonical_comparison_unavailable" in reasons
        for reasons in first["audit"]["excluded_record_ids"].values()
    )
    assert first["audit"]["treebanks"]["sdh"]["source_sentences"] == 15
    assert first["audit"]["treebanks"]["sdh"]["accepted"] == 12
    assert first["audit"]["cross_original_split_duplicate_groups"] > 0
    assert first["audit"]["cross_source"]["parme"]["affected_ud_sentences"] > 0
    assert first["audit"]["cross_source"]["parme"]["sdh_script_distribution"] == {
        "latin": 1
    }
    assert first["audit"]["legacy_ud"]["files"]["kmr_kurmanji-ud-train.conllu"][
        "byte_identical"
    ]
    assert (
        validate_benchmark(
            output / "ud_kurdish.jsonl", output / "ud_kurdish.provenance.jsonl"
        )["records"]
        == 20
    )
    if first["audit"]["strict_external_count"]:
        assert (
            validate_benchmark(
                output / "ud_kurdish.strict.jsonl",
                output / "ud_kurdish.strict.provenance.jsonl",
            )["records"]
            == first["audit"]["strict_external_count"]
        )
    hashes = first["manifest"]["output_hashes"]
    second = ud.build(
        ("kmr", "sdh"),
        raw_dir=raw,
        output_dir=output,
        canonical_dir=canonical,
        legacy_dir=legacy,
    )
    assert second["manifest"]["output_hashes"] == hashes
    assert all(
        row["evaluation_partition"] == "external_test"
        for row in map(
            json.loads,
            (output / "ud_kurdish.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    )


def test_corrupt_cached_archive_fails_closed(
    tmp_path: Path,
) -> None:
    path = ud.archive_path("sdh", tmp_path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ud.acquire(("sdh",), raw_dir=tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        ud.build(
            ("sdh",),
            raw_dir=tmp_path,
            output_dir=tmp_path / "out",
            canonical_dir=tmp_path,
            legacy_dir=tmp_path,
        )


def test_valid_cached_archive_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = ud.archive_path("sdh", tmp_path)
    digest = _synthetic_archive(archive, "sdh")
    manifest = replace(ud.manifests_for(("sdh",))["sdh"], checksum_sha256=digest)
    monkeypatch.setattr(ud, "manifests_for", lambda languages: {"sdh": manifest})

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be used for a valid cache")

    receipt = ud.acquire(("sdh",), raw_dir=tmp_path, opener=no_network)
    assert receipt["sdh"]["reused"] is True
    assert receipt["sdh"]["sha256"] == digest

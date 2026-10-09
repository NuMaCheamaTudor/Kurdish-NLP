"""Offline acquisition metadata, policy, provenance and CLI tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kurdish_nlp.langid.acquisition.licenses import (
    LICENSE_RIGHTS,
    LicenseId,
    LicenseSpec,
)
from kurdish_nlp.langid.acquisition.manifests import (
    SourceIndex,
    SourceManifest,
    load_index,
    load_indexed_manifests,
    load_manifest,
    save_index,
    save_manifest,
)
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.provenance import (
    ProvenanceRecord,
    Transformation,
    load_sidecars,
    save_sidecars,
    validate_linkage,
)
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import DatasetRecord

FIXTURES = Path(__file__).parent / "fixtures" / "langid" / "manifests"


def fixture(name: str) -> SourceManifest:
    return load_manifest(FIXTURES / f"{name}.example.v1.json")


def codes(findings: object) -> set[str]:
    return {finding.code for finding in findings}


@pytest.mark.parametrize(
    "name", ["parme", "tatoeba", "wikimedia", "unverified", "southern-asr"]
)
def test_fixture_manifests_load_and_round_trip(name: str, tmp_path: Path) -> None:
    original = fixture(name)
    output = tmp_path / "source.json"
    save_manifest(original, output)
    assert load_manifest(output) == original
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert save_and_read_again(original, tmp_path) == output.read_bytes()


def save_and_read_again(manifest: SourceManifest, tmp_path: Path) -> bytes:
    second = tmp_path / "second.json"
    save_manifest(manifest, second)
    return second.read_bytes()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "unsupported schema_version"),
        ({"source_id": "Bad ID"}, "source_id"),
        ({"source_url": "javascript:alert(1)"}, "source_url"),
        (
            {"retrieved_at": "2026-10-07T12:00:00", "checksum_sha256": "a" * 64},
            "timezone",
        ),
        (
            {"retrieved_at": "2026-10-07T12:00:00Z", "checksum_sha256": "abc"},
            "checksum_sha256",
        ),
        ({"retrieved_at": "2026-10-07T12:00:00Z"}, "checksum_sha256"),
        ({"allowed_roles": ["not_a_role"]}, "allowed_roles"),
        ({"provenance_status": "maybe"}, "provenance_status"),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_manifest_validation(change: dict[str, object], message: str) -> None:
    value = fixture("parme").to_dict()
    value.update(change)
    with pytest.raises((ValueError, TypeError), match=message):
        SourceManifest.from_mapping(value)


def test_license_metadata_must_be_verified_and_consistent() -> None:
    value = fixture("parme").to_dict()
    value["license"]["verification_url"] = None
    with pytest.raises(ValueError, match="verification_url"):
        SourceManifest.from_mapping(value)
    value = fixture("parme").to_dict()
    value["license"]["commercial_use"] = "prohibited"
    with pytest.raises(ValueError, match="contradict"):
        SourceManifest.from_mapping(value)
    value = fixture("parme").to_dict()
    value["license"]["extra"] = "ignored?"
    with pytest.raises(ValueError, match="unknown fields"):
        SourceManifest.from_mapping(value)


@pytest.mark.parametrize("identifier", list(LicenseId))
def test_each_license_preset(identifier: LicenseId) -> None:
    url = None if identifier is LicenseId.UNKNOWN else "https://example.org/license"
    spec = LicenseSpec.for_identifier(identifier, url)
    assert LicenseSpec.from_mapping(spec.to_dict()) == spec
    assert (
        spec.commercial_use,
        spec.redistribution,
        spec.derivatives,
        spec.attribution_required,
        spec.share_alike_required,
    ) == LICENSE_RIGHTS[identifier]


@pytest.mark.parametrize(
    ("name", "allowed", "reason", "warnings"),
    [
        ("parme", True, None, {"attribution_required", "redistribution_conditional"}),
        ("tatoeba", True, None, {"attribution_required", "provenance_caveats"}),
        ("wikimedia", True, None, {"attribution_required", "share_alike_required"}),
        ("unverified", False, "license_unverified", set()),
        ("southern-asr", False, "commercial_use_not_allowed", set()),
    ],
)
def test_commercial_policy(
    name: str, allowed: bool, reason: str | None, warnings: set[str]
) -> None:
    decision = evaluate(fixture(name), role="training")
    assert decision.allowed is allowed
    if reason:
        assert reason in codes(decision.reasons)
    assert warnings <= codes(decision.warnings)
    assert decision.to_dict()["profile"] == "commercial"


def test_nd_block_and_research_nc_exception() -> None:
    nd = evaluate(fixture("southern-asr"), role="training", profile="research")
    assert not nd.allowed
    assert "derivatives_prohibited" in codes(nd.reasons)

    nc = replace(
        fixture("parme"),
        license=LicenseSpec.for_identifier(
            LicenseId.CC_BY_NC_SA_4,
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ),
    )
    assert not evaluate(nc, role="training", profile="commercial").allowed
    research = evaluate(nc, role="training", profile="research")
    assert research.allowed
    assert "research_only" in codes(research.warnings)


def test_role_restriction_and_unverified_research_block() -> None:
    restricted = replace(fixture("parme"), allowed_roles=("benchmark",))
    decision = evaluate(restricted, role="training")
    assert not decision.allowed
    assert "role_not_allowed" in codes(decision.reasons)
    assert not evaluate(
        fixture("unverified"), role="training", profile="research"
    ).allowed
    assert evaluate(fixture("unverified"), role="provenance_only").allowed


def test_index_is_versioned_and_round_trips(tmp_path: Path) -> None:
    repo_index = load_index(
        Path(__file__).parents[1] / "configs" / "langid" / "sources.v1.json"
    )
    assert repo_index == SourceIndex(
        1,
        (
            "sources/parme.v1.json",
            "sources/tatoeba.v1.json",
            "sources/wikimedia-ckb.v1.json",
            "sources/wikimedia-kmr.v1.json",
            "sources/wikimedia-ar.v1.json",
            "sources/wikimedia-fa.v1.json",
            "sources/wikimedia-tr.v1.json",
            "sources/wikimedia-en.v1.json",
            "sources/ud-sdh-garrusi-v2.18.v1.json",
            "sources/ud-kmr-kurmanji-v2.18.v1.json",
        ),
    )
    index = SourceIndex(1, ("sources/parme.v1.json",))
    output = tmp_path / "index.json"
    save_index(index, output)
    assert load_index(output) == index
    with pytest.raises(ValueError, match="relative"):
        SourceIndex(1, ("../outside.json",))
    approved = load_indexed_manifests(
        Path(__file__).parents[1] / "configs" / "langid" / "sources.v1.json"
    )
    assert [item.source_id for item in approved] == [
        "parme",
        "tatoeba",
        "wikimedia-ckb",
        "wikimedia-kmr",
        "wikimedia-ar",
        "wikimedia-fa",
        "wikimedia-tr",
        "wikimedia-en",
        "ud-sdh-garrusi-v2-18",
        "ud-kmr-kurmanji-v2-18",
    ]
    sources = tmp_path / "sources"
    sources.mkdir()
    save_manifest(fixture("parme"), sources / "parme.v1.json")
    save_index(index, output)
    assert load_indexed_manifests(output) == (fixture("parme"),)


def test_provenance_sidecar_linkage_and_round_trip(tmp_path: Path) -> None:
    canonical = DatasetRecord.from_mapping(
        {
            "id": "sdh-1",
            "text": "sample",
            "label": "sdh",
            "source": "parme",
            "domain": "translation",
            "document_id": "alignment-1",
            "license": "MIT",
            "split": "train",
        }
    )
    provenance = ProvenanceRecord(
        schema_version=1,
        canonical_record_id="sdh-1",
        source_id="parme",
        acquisition_manifest_version="1",
        original_record_id="row-1",
        original_document_id="alignment-1",
        original_url="https://example.org/row/1",
        contributor_id="translator-7",
        author_id="author-3",
        translator_id="translator-7",
        language_variety="Kalhori",
        script="Arab",
        original_split="train",
        original_license="MIT",
        original_text_sha256="a" * 64,
        transformations=(
            Transformation(
                "normalize",
                "1",
                "2026-10-07T12:00:00+03:00",
                {"unicode": "NFC", "preserve_zwnj": True},
                "raw:row-1",
                "canonical:sdh-1",
            ),
        ),
    )
    output = tmp_path / "provenance.jsonl"
    save_sidecars([provenance], output)
    loaded = load_sidecars(output)
    assert loaded == [provenance]
    assert loaded[0].to_dict()["transformations"][0]["parameters"]["unicode"] == "NFC"
    validate_linkage([canonical], loaded)
    with pytest.raises(ValueError, match="match canonical"):
        validate_linkage(
            [canonical], [replace(provenance, canonical_record_id="other")]
        )


def test_provenance_rejects_code_and_unknown_fields() -> None:
    with pytest.raises((TypeError, ValueError), match="parameters"):
        Transformation("bad", "1", parameters={"script": object()})
    with pytest.raises(ValueError, match="unknown fields"):
        ProvenanceRecord.from_mapping(
            {
                **ProvenanceRecord(1, "id", "parme", "1").to_dict(),
                "surprise": "x",
            }
        )


def test_provenance_optional_fields_can_be_absent() -> None:
    minimal = ProvenanceRecord.from_mapping(
        {
            "schema_version": 1,
            "canonical_record_id": "id-1",
            "source_id": "parme",
            "acquisition_manifest_version": "1",
            "transformations": [{"name": "extract", "version": "1"}],
        }
    )
    assert minimal.author_id is None
    assert minimal.transformations[0].parameters == {}
    assert (
        ProvenanceRecord.from_mapping(
            {
                "schema_version": 1,
                "canonical_record_id": "id-2",
                "source_id": "parme",
                "acquisition_manifest_version": "1",
            }
        ).transformations
        == ()
    )


def test_cli_manifest_commands(capsys: pytest.CaptureFixture[str]) -> None:
    parme = str(FIXTURES / "parme.example.v1.json")
    unverified = str(FIXTURES / "unverified.example.v1.json")
    assert main(["manifest", "validate", parme]) == 0
    assert "valid source manifest" in capsys.readouterr().out
    assert main(["manifest", "show", parme]) == 0
    assert json.loads(capsys.readouterr().out)["source_id"] == "parme"
    assert main(["manifest", "policy-check", parme, "--role", "training"]) == 0
    assert json.loads(capsys.readouterr().out)["allowed"] is True
    assert (
        main(
            [
                "manifest",
                "policy-check",
                unverified,
                "--role",
                "training",
                "--profile",
                "research",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["allowed"] is False


def test_cli_research_nc_and_existing_dataset_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nc = replace(
        fixture("parme"),
        license=LicenseSpec.for_identifier(
            LicenseId.CC_BY_NC_SA_4,
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ),
    )
    path = tmp_path / "nc.json"
    save_manifest(nc, path)
    assert (
        main(
            [
                "manifest",
                "policy-check",
                str(path),
                "--role",
                "training",
                "--profile",
                "research",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["allowed"] is True
    assert (
        main(
            [
                "validate",
                str(Path(__file__).parent / "fixtures" / "langid" / "synthetic.jsonl"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["record_count"] == 7


def test_manifest_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_manifest(path)


def test_cli_reports_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":2}', encoding="utf-8")
    assert main(["manifest", "validate", str(path)]) == 1
    assert "missing fields" in capsys.readouterr().out

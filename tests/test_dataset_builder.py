"""Offline end-to-end tests of the dataset-v1 builder on a synthetic fixture world."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from conftest import UD_TEXTS, write_config

from kurdish_nlp.langid.acquisition.licenses import LicenseSpec
from kurdish_nlp.langid.acquisition.manifests import load_manifest, save_manifest
from kurdish_nlp.langid.acquisition.provenance import (
    load_sidecars,
    validate_parallel_jsonl_linkage,
)
from kurdish_nlp.langid.build import (
    BuildError,
    BuildOptions,
    build_dataset,
    load_config,
)
from kurdish_nlp.langid.build.config import BuildConfigError
from kurdish_nlp.langid.build.export import REPRODUCIBLE_FILES
from kurdish_nlp.langid.build.sources import SourceVerificationError
from kurdish_nlp.langid.build.textkeys import loose_key
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import validate_jsonl
from kurdish_nlp.langid.normalization import normalize_text

CONFIG = "configs/langid/dataset-test.json"
OUTPUT = "data/langid/builds/dataset-test"


def _build(root: Path, config: str | Path = CONFIG, **options: object) -> dict:
    return build_dataset(
        root / config, BuildOptions(base_dir=root, workers=1, **options)
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def built(build_world: Path) -> tuple[Path, dict]:
    return build_world, _build(build_world)


# ------------------------------------------------------------ verification
def test_dry_run_verifies_without_writing(build_world: Path) -> None:
    result = _build(build_world, dry_run=True)
    assert result["dry_run"] is True
    assert result["plan"]["verification"] == "passed"
    assert {item["family"] for item in result["plan"]["sources"]} == {
        "parme",
        "tatoeba",
        "wikimedia",
    }
    assert not (build_world / OUTPUT).exists()
    assert not (build_world / "data/langid/cache/dataset-test").exists()


def test_missing_source_artifact_fails(build_world: Path) -> None:
    (build_world / "data/langid/processed/wikimedia.jsonl").unlink()
    with pytest.raises(SourceVerificationError, match="canonical export not found"):
        _build(build_world, dry_run=True)


def test_invalid_checksum_fails(build_world: Path) -> None:
    with (build_world / "data/langid/processed/tatoeba.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")
    with pytest.raises(SourceVerificationError, match="checksum mismatch"):
        _build(build_world, dry_run=True)


def test_tampered_acquisition_receipt_fails(build_world: Path) -> None:
    (build_world / "data/langid/raw/tatoeba/links.tar.bz2").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _build(build_world, dry_run=True)


def _replace_parme_manifest(root: Path, **changes: object) -> None:
    path = root / "configs/langid/sources/parme.v1.json"
    from dataclasses import replace

    save_manifest(replace(load_manifest(path), **changes), path)


def test_noncommercial_license_is_blocked(build_world: Path) -> None:
    _replace_parme_manifest(
        build_world,
        license=LicenseSpec.for_identifier(
            "CC-BY-NC-SA-4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"
        ),
    )
    with pytest.raises(SourceVerificationError, match="commercial_use_not_allowed"):
        _build(build_world, dry_run=True)


def test_unverified_provenance_is_blocked(build_world: Path) -> None:
    _replace_parme_manifest(build_world, provenance_status="unverified")
    with pytest.raises(SourceVerificationError, match="provenance_unverified"):
        _build(build_world, dry_run=True)


def test_unsupported_source_role_is_blocked(build_world: Path) -> None:
    _replace_parme_manifest(
        build_world, allowed_roles=("training", "external_evaluation")
    )
    with pytest.raises(SourceVerificationError, match="role_not_allowed"):
        _build(build_world, dry_run=True)


def test_benchmark_manifest_cannot_be_a_training_source(build_world: Path) -> None:
    config = json.loads((build_world / CONFIG).read_text(encoding="utf-8"))
    config["sources"][0]["manifests"].append(
        "configs/langid/sources/ud-kmr-kurmanji-v2.18.v1.json"
    )
    (build_world / CONFIG).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SourceVerificationError, match="protected benchmark source"):
        _build(build_world, dry_run=True)


def test_benchmark_path_cannot_be_a_training_artifact(build_world: Path) -> None:
    config = json.loads((build_world / CONFIG).read_text(encoding="utf-8"))
    benchmark = config["benchmark_protection"]["benchmarks"][0]["records"]
    config["sources"][1]["canonical"] = benchmark
    (build_world / CONFIG).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SourceVerificationError, match="benchmark path"):
        _build(build_world, dry_run=True)


def test_config_rejects_und_and_unknown_settings(build_world: Path) -> None:
    config = json.loads((build_world / CONFIG).read_text(encoding="utf-8"))
    config["labels"] = [*config["labels"], "und"]
    (build_world / "bad.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BuildConfigError, match="und is never a training class"):
        load_config(build_world / "bad.json")
    config = json.loads((build_world / CONFIG).read_text(encoding="utf-8"))
    config["sampling"]["surprise"] = 1
    (build_world / "bad.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(build_world / "bad.json")


# ------------------------------------------------------------- full build
def test_build_outputs_validate_and_link_provenance(built: tuple[Path, dict]) -> None:
    root, result = built
    output = root / OUTPUT
    for name in (*REPRODUCIBLE_FILES, "manifest.json", "build_run.json"):
        assert (output / name).is_file(), name
    combined = validate_jsonl(output / "dataset.jsonl")
    assert combined.text_split_leakage == ()
    assert combined.document_split_leakage == ()
    for split in ("train", "dev", "test"):
        assert set(validate_jsonl(output / f"{split}.jsonl").split_distribution) == {
            split
        }
    assert (
        validate_parallel_jsonl_linkage(
            output / "dataset.jsonl", output / "provenance.jsonl"
        )
        == combined.record_count
    )
    sidecars = load_sidecars(output / "provenance.jsonl")
    assert all(item.transformations[-1].name == "dataset_build" for item in sidecars)
    assert all(
        item.transformations[-1].parameters["review"] == {"status": "unreviewed"}
        for item in sidecars
    )
    rows = _rows(output / "dataset.jsonl")
    assert set(rows[0]) == {
        "id",
        "text",
        "label",
        "source",
        "domain",
        "document_id",
        "license",
        "split",
    }
    assert result["validation"]["benchmark_recheck_matches"] == 0
    assert result["run"]["lock_status"] == "written"
    assert (root / "configs/langid/dataset-test.lock.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_timestamp"] is None
    assert set(manifest["outputs"]) == set(REPRODUCIBLE_FILES)


def test_benchmark_overlaps_are_quarantined_and_never_exported(
    built: tuple[Path, dict],
) -> None:
    root, _ = built
    output = root / OUTPUT
    exported = _rows(output / "dataset.jsonl")
    ud_exact = {normalize_text(text) for _, text, _ in UD_TEXTS.values()}
    ud_loose = {loose_key(text) for _, text, _ in UD_TEXTS.values()}
    assert not any(
        row["text"] in ud_exact or loose_key(row["text"]) in ud_loose
        for row in exported
    )
    assert not any(row["id"].startswith("ud:") for row in exported)
    quarantined = {row["record_id"]: row for row in _rows(output / "quarantined.jsonl")}
    types = {
        record_id: {match["match_type"] for match in row["benchmark_matches"]}
        for record_id, row in quarantined.items()
        if row["benchmark_matches"]
    }
    assert types["tatoeba:kmr:707"] == {"exact_normalized"}
    assert types["tatoeba:kmr:708"] == {"loose_normalized"}
    assert types["tatoeba:kmr:709"] == {"near_jaccard"}
    assert any(
        kinds == {"contained_sentence"} and key.startswith("wikimedia:kuwiki")
        for key, kinds in types.items()
    )
    assert all("benchmark_overlap" in quarantined[key]["reasons"] for key in types)


def test_duplicates_ambiguity_and_canonical_selection(built: tuple[Path, dict]) -> None:
    root, _ = built
    output = root / OUTPUT
    exported = {row["id"]: row for row in _rows(output / "dataset.jsonl")}
    texts = {row["text"] for row in exported.values()}
    # Loose same-label variants collapse to one canonical record.
    assert ("tatoeba:en:130" in exported) != ("tatoeba:en:131" in exported)
    # PARME is preferred over Tatoeba for an identical sdh text.
    assert "tatoeba:sdh:604" not in exported
    assert "ئەیە کتاوەگەی منە." in texts
    # Wikimedia is preferred over Tatoeba for an identical ckb text.
    assert "tatoeba:ckb:508" not in exported
    assert "من دەچمە بازاڕ بۆ کڕینی نان." in texts
    # Cross-label strings are withheld and preserved for an ambiguity benchmark.
    assert "القاهرة مدينة كبيرة جدا." not in texts
    assert "tatoeba:ckb:509" not in exported and "tatoeba:sdh:605" not in exported
    ambiguous = _rows(output / "ambiguous.jsonl")
    label_sets = sorted(tuple(row["labels"]) for row in ambiguous)
    assert label_sets == [("ar", "fa"), ("ckb", "sdh")]
    duplicates = _rows(output / "duplicates.jsonl")
    sdh_cluster = next(
        row
        for row in duplicates
        if "tatoeba:sdh:604" in {m["record_id"] for m in row["members"]}
    )
    assert sdh_cluster["kind"] == "same_label"
    assert sdh_cluster["canonical_record_id"].startswith("parme:sdh:")
    assert {m["source_id"] for m in sdh_cluster["members"]} == {"parme", "tatoeba"}
    # Records below the runtime minimum letter count are excluded with a reason.
    reasons = {
        row["record_id"]: row["reasons"] for row in _rows(output / "quarantined.jsonl")
    }
    assert reasons["tatoeba:en:133"] == ["below_min_letters"]


def test_groups_never_cross_splits(built: tuple[Path, dict]) -> None:
    root, _ = built
    output = root / OUTPUT
    rows = _rows(output / "dataset.jsonl")
    sidecars = {
        item.canonical_record_id: item
        for item in load_sidecars(output / "provenance.jsonl")
    }
    by_source_document: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = sidecars[row["id"]].transformations[-1].parameters["source_export"]
        by_source_document[source["document_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in by_source_document.values())
    split = {row["id"]: row["split"] for row in rows}
    group = {row["id"]: row["document_id"] for row in rows}
    # The giant Tatoeba component stays whole and goes to train.
    giant = [
        record_id
        for record_id in split
        if record_id.startswith("tatoeba:")
        and record_id.split(":")[2] in {str(i) for i in range(1, 11)}
    ]
    assert giant and {split[record_id] for record_id in giant} == {"train"}
    assert len({group[record_id] for record_id in giant}) == 1
    # Multi-language group: sdh 600 is linked to en 100 and sdh records are kept whole.
    if "tatoeba:en:100" in split:
        assert split["tatoeba:en:100"] == split["tatoeba:sdh:600"]
    # Sentence containment joins the Tatoeba sentence with the Wikipedia segment.
    contained = next(
        record_id
        for record_id, row in zip(split, rows)
        if row["text"].endswith("The weather is nice today.")
        and record_id.startswith("wikimedia")
    )
    assert group[contained] == group["tatoeba:en:132"]
    near = _rows(output / "near_duplicates.jsonl")
    assert all(len(set(edge["final_splits"])) == 1 for edge in near)


def test_sampling_caps_and_scarce_class_protection(built: tuple[Path, dict]) -> None:
    root, result = built
    counts = result["counts"]
    assert counts["en"]["total"] == 20 and counts["tr"]["total"] == 20
    assert counts["ar"]["total"] <= 8
    audit = json.loads((root / OUTPUT / "audit.json").read_text(encoding="utf-8"))
    sampling = audit["sampling"]["per_label"]
    for label in ("ckb", "kmr", "sdh", "fa"):
        assert sampling[label]["mode"] == "keep_all_eligible"
        assert sampling[label]["selected_total"] == sampling[label]["eligible_total"]
    en = sampling["en"]["sources"]["tatoeba"]
    assert en["contributor_cap_records"] == int(0.25 * en["quota"])
    assert en["slices_deferred_by_contributor_cap"] > 0
    assert (
        sampling["en"]["selected_by_source"]["wikimedia-en"]
        == sampling["en"]["eligible_by_source"]["wikimedia-en"]
    )
    contributors = audit["contributors"]["final_by_label"]["en"]["top"]
    big = next(
        item for item in contributors if item["contributor"] == "big_contributor"
    )
    assert big["records"] < 20  # the eligible pool holds 20 of its sentences
    assert audit["validation"]["sampling_caps_respected"] is True


def test_rebuild_is_byte_identical_and_lock_matches(built: tuple[Path, dict]) -> None:
    root, first = built
    second = _build(root, output_dir="data/langid/builds/rebuild")
    assert second["run"]["lock_status"] == "matched"
    for name in REPRODUCIBLE_FILES:
        assert first["outputs"][name]["sha256"] == second["outputs"][name]["sha256"], (
            name
        )


def test_source_order_does_not_change_data(build_world: Path) -> None:
    first = _build(build_world)
    config = json.loads((build_world / CONFIG).read_text(encoding="utf-8"))
    config["sources"].reverse()
    config["sources"][0]["manifests"].reverse()
    config["outputs"]["lock_file"] = "configs/langid/reordered.lock.json"
    config["outputs"]["directory"] = "data/langid/builds/reordered"
    path = build_world / "configs/langid/reordered.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    second = _build(build_world, "configs/langid/reordered.json")
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset.jsonl",
        "duplicates.jsonl",
        "near_duplicates.jsonl",
        "quarantined.jsonl",
        "ambiguous.jsonl",
        "review_queue.jsonl",
    ):
        assert first["outputs"][name]["sha256"] == second["outputs"][name]["sha256"], (
            name
        )


def test_changed_config_cannot_silently_overwrite_frozen_release(
    built: tuple[Path, dict],
) -> None:
    root, _ = built
    write_config(root, seed=7)
    with pytest.raises(BuildError, match="bump dataset_version"):
        _build(root, dry_run=True)


def test_review_sidecar_relabels_and_rejects_through_explicit_rebuild(
    build_world: Path,
) -> None:
    review = build_world / "reviews.jsonl"
    common = {
        "schema_version": 1,
        "review_version": "1",
        "reviewer_id": "synthetic-test-reviewer",
        "reviewed_at": "2026-11-01T10:00:00+00:00",
        "quality_status": "ok",
        "contamination_flags": [],
        "notes": "synthetic test decision",
    }
    review.write_text(
        json.dumps(
            {
                **common,
                "canonical_record_id": "tatoeba:kmr:700",
                "decision": "relabel",
                "corrected_label": "ckb",
            }
        )
        + "\n"
        + json.dumps(
            {
                **common,
                "canonical_record_id": "tatoeba:kmr:701",
                "decision": "reject",
                "corrected_label": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import hashlib

    write_config(
        build_world,
        review__sidecars=[
            {
                "path": "reviews.jsonl",
                "sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
            }
        ],
    )
    _build(build_world)
    output = build_world / OUTPUT
    rows = {row["id"]: row for row in _rows(output / "dataset.jsonl")}
    assert rows["tatoeba:kmr:700"]["label"] == "ckb"
    assert "tatoeba:kmr:701" not in rows
    sidecar = next(
        item
        for item in load_sidecars(output / "provenance.jsonl")
        if item.canonical_record_id == "tatoeba:kmr:700"
    )
    parameters = sidecar.transformations[-1].parameters
    assert parameters["review"]["decision"] == "relabel"
    assert parameters["source_export"]["label"] == "kmr"
    quarantined = {row["record_id"]: row for row in _rows(output / "quarantined.jsonl")}
    assert quarantined["tatoeba:kmr:701"]["reasons"] == ["human_review_reject"]


def test_cli_dry_run_build_and_errors(
    build_world: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(build_world)
    assert (
        main(["build", "dataset-test", "--dry-run", "--workers", "1", "--quiet"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["plan"]["verification"] == "passed"
    assert main(["build", "dataset-other", "--config", CONFIG, "--quiet"]) == 1
    assert "does not configure dataset" in capsys.readouterr().out
    assert main(["build", "dataset-missing", "--quiet"]) == 1
    assert "not found" in capsys.readouterr().out
    assert main(["build", "dataset-test", "--workers", "1", "--quiet"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation"]["global_group_cross_split_leakage"] == 0
    assert main(["validate", f"{OUTPUT}/dataset.jsonl"]) == 0


def test_conflicting_source_splits_are_reconciled_and_order_is_stable(
    built: tuple[Path, dict],
) -> None:
    root, _ = built
    output = root / OUTPUT
    rows = _rows(output / "dataset.jsonl")
    sidecars = load_sidecars(output / "provenance.jsonl")
    source_splits: dict[str, set[str]] = defaultdict(set)
    final_splits: dict[str, set[str]] = defaultdict(set)
    for row, sidecar in zip(rows, sidecars, strict=True):
        assert sidecar.canonical_record_id == row["id"]
        parameters = sidecar.transformations[-1].parameters
        source_splits[row["document_id"]].add(parameters["source_export"]["split"])
        final_splits[row["document_id"]].add(row["split"])
    conflicting = [group for group, splits in source_splits.items() if len(splits) > 1]
    assert conflicting, "fixture must contain groups whose source exports disagreed"
    assert all(len(final_splits[group]) == 1 for group in conflicting)
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["splits"]["final_groups_with_conflicting_source_splits"] == len(
        conflicting
    )
    for split in ("train", "dev", "test"):
        ids = [row["id"] for row in _rows(output / f"{split}.jsonl")]
        assert ids == sorted(ids)
    # Original source IDs are preserved as stable canonical IDs.
    assert {row["id"].split(":")[0] for row in rows} == {
        "parme",
        "tatoeba",
        "wikimedia",
    }

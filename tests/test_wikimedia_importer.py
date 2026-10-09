"""Entirely offline contracts for Wikimedia acquisition, parsing and import."""

from __future__ import annotations

import bz2
import hashlib
import json
from collections import Counter
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from kurdish_nlp.langid.acquisition.importers import wikimedia, wikimedia_acquire
from kurdish_nlp.langid.acquisition.importers.wikimedia_extract import (
    extract_segments,
    script_flags,
)
from kurdish_nlp.langid.acquisition.manifests import load_manifest
from kurdish_nlp.langid.acquisition.provenance import validate_parallel_jsonl_linkage
from kurdish_nlp.langid.cli import main
from kurdish_nlp.langid.dataset import validate_jsonl

FIXTURE = Path(__file__).parent / "fixtures/langid/wikimedia/pages.json"


def test_reviewed_manifests_and_commercial_policy() -> None:
    plan = wikimedia.plan_acquisition(
        ("ckb", "kmr", "ar", "fa", "tr", "en"), max_pages=20
    )
    assert set(plan["sources"]) == set(wikimedia.PROJECTS)
    assert all(item["policy"]["allowed"] for item in plan["sources"].values())
    for label in wikimedia.PROJECTS:
        manifest = load_manifest(
            Path("configs/langid/sources") / f"wikimedia-{label}.v1.json"
        )
        assert manifest.license.attribution_required
        assert manifest.license.share_alike_required
    with pytest.raises(ValueError, match="official project URL"):
        manifest = wikimedia.manifests_for(("ar",))["ar"]
        from dataclasses import replace

        wikimedia._check_manifest(
            replace(manifest, source_url="https://example.org/w/api.php"), "ar"
        )
    with pytest.raises(ValueError, match="unsupported Wikimedia"):
        wikimedia.selected_languages(("sdh",))


def test_extraction_removes_markup_and_preserves_script_diagnostics() -> None:
    pages = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sorani, reason = extract_segments(pages[0]["wikitext"], label="ckb")
    assert reason is None and len(sorani) == 1
    assert "citation" not in sorani[0].text
    assert "table cell" not in sorani[0].text
    assert "Infobox" not in sorani[0].text
    kurmanji, reason = extract_segments(pages[1]["wikitext"], label="kmr")
    assert reason is None and len(kurmanji) == 1
    assert "References" not in kurmanji[0].text
    assert "mixed_script" in script_flags(pages[2]["wikitext"], "kmr")
    assert extract_segments(pages[3]["wikitext"], label="en")[1] == "redirect"
    assert extract_segments(pages[4]["wikitext"], label="en")[1] == "disambiguation"
    assert extract_segments(pages[5]["wikitext"], label="en")[1] == "empty"
    assert extract_segments(
        "This is a sufficiently long sentence.", label="en", mode="sentence"
    )[0]
    assert extract_segments("متن فارسی برای بررسی این آزمایش است.", label="fa")[0]
    assert extract_segments("هذا نص عربي صالح للتجربة والتحليل.", label="ar")[0]
    contaminated = (
        "{{Infobox ship\n| name = vessel\n\n| other = [[File:ship.jpg|caption]]\n}} "
        "این یک مقاله فارسی درباره تاریخ کشتی و دریانوردی است.\n"
        "<gallery>\nFile:ship.jpg|gallery caption\n</gallery>\n"
        "این بخش نیز متن طبیعی و مناسب برای آزمایش است."
    )
    cleaned, reason = extract_segments(contaminated, label="fa")
    assert reason is None
    assert all(
        "Infobox" not in item.text and "ship.jpg" not in item.text for item in cleaned
    )
    assert all("gallery caption" not in item.text for item in cleaned)
    rejected: Counter[str] = Counter()
    fragments, _ = extract_segments(
        "A valid natural language paragraph with enough words. "
        "[[File: broken markup|caption with enough words.",
        label="en",
        mode="sentence",
        rejections=rejected,
    )
    assert len(fragments) == 1
    assert rejected["residual_markup"] == 1


def _write_api_fixture(raw_dir: Path, label: str, pages: list[dict]) -> None:
    manifest = wikimedia.manifests_for((label,))[label]
    project = wikimedia.PROJECTS[label][0]
    directory = raw_dir / project
    wikimedia_acquire._write_pages(directory / "pages.jsonl", pages)
    wikimedia_acquire._receipt(
        manifest,
        project,
        directory,
        method="revision-pinned-action-api",
        pages=pages,
        source={
            "url": manifest.source_url,
            "cached_page_bytes": 100,
            "network_transfer_bytes": None,
        },
        sampling={
            "method": "fixture",
            "seed": 42,
            "frame_size": len(pages),
            "target_pages": len(pages),
        },
    )


def test_offline_import_dedup_linkage_and_reproducibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ar_pages = [
        {
            **fixture[0],
            "page_id": "101",
            "wikitext": "هذا مقال عربي يتحدث عن تاريخ مدينة قديمة ومعلومات كثيرة.",
        },
        {
            **fixture[1],
            "page_id": "102",
            "wikitext": "هذا مقال عربي يتحدث عن تاريخ مدينة قديمة ومعلومات كثيرة.",
        },
        {
            **fixture[2],
            "page_id": "103",
            "wikitext": "تقدم هذه الصفحة معلومات مختلفة عن اللغة العربية وتاريخها الطويل.",
        },
    ]
    fa_pages = [
        {
            **fixture[0],
            "page_id": "201",
            "wikitext": "این مقاله درباره تاریخ یک شهر کهن و ویژگی‌های آن است.",
        },
        {
            **fixture[1],
            "page_id": "202",
            "wikitext": "هذا مقال عربي يتحدث عن تاريخ مدينة قديمة ومعلومات كثيرة.",
        },
    ]
    raw = tmp_path / "raw"
    output = tmp_path / "out"
    _write_api_fixture(raw, "ar", ar_pages)
    _write_api_fixture(raw, "fa", fa_pages)
    monkeypatch.setattr(
        wikimedia_acquire,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "offline import attempted network access"
        ),
    )
    assert (
        main(
            [
                "import",
                "wikimedia",
                "--languages",
                "ar",
                "fa",
                "--raw-dir",
                str(raw),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    audit = json.loads((output / "wikimedia.audit.json").read_text(encoding="utf-8"))
    assert audit["segments_extracted"] == 5
    assert audit["segments_accepted"] == 2
    assert audit["rejection_counts"]["cross_label_identical_text"] == 3
    assert audit["rejected_candidate_provenance_count"] == 3
    rejected = [
        json.loads(line)
        for line in (output / "wikimedia.rejections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rejected) == 3
    assert all(item["reason"] == "cross_label_identical_text" for item in rejected)
    assert validate_jsonl(output / "wikimedia.jsonl").record_count == 2
    assert (
        validate_parallel_jsonl_linkage(
            output / "wikimedia.jsonl", output / "wikimedia.provenance.jsonl"
        )
        == 2
    )
    first = (audit["canonical_sha256"], audit["provenance_sha256"])
    assert (
        wikimedia.build(("fa", "ar"), raw_dir=raw, output_dir=output)[
            "canonical_sha256"
        ]
        == first[0]
    )
    repeated = json.loads((output / "wikimedia.audit.json").read_text(encoding="utf-8"))
    assert (repeated["canonical_sha256"], repeated["provenance_sha256"]) == first
    with pytest.raises(ValueError, match="checksum"):
        (raw / "arwiki/pages.jsonl").write_text("tampered\n", encoding="utf-8")
        wikimedia.build(("ar",), raw_dir=raw, output_dir=output)


def test_dump_page_sample_is_order_independent(tmp_path: Path) -> None:
    def archive(path: Path, ids: list[int]) -> None:
        pages = "".join(
            f"<page><title>Page {n}</title><ns>0</ns><id>{n}</id>"
            f"<revision><id>{n + 100}</id><timestamp>2026-10-01T00:00:00Z</timestamp>"
            f"<text>Article text {n}.</text></revision></page>"
            for n in ids
        )
        path.write_bytes(bz2.compress(f"<mediawiki>{pages}</mediawiki>".encode()))

    first, second = tmp_path / "first.bz2", tmp_path / "second.bz2"
    archive(first, list(range(1, 21)))
    archive(second, list(range(20, 0, -1)))
    assert wikimedia_acquire.select_dump_page_ids(
        first, seed=7, max_pages=5
    ) == wikimedia_acquire.select_dump_page_ids(second, seed=7, max_pages=5)


def test_retry_after_429_and_client_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Response:
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return b"{}"

    def fake_open(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            headers = Message()
            headers["Retry-After"] = "2"
            raise HTTPError(request.full_url, 429, "throttle", headers, None)
        return Response()

    slept = []
    monkeypatch.setattr(wikimedia_acquire, "urlopen", fake_open)
    monkeypatch.setattr(wikimedia_acquire.time, "sleep", slept.append)
    client = wikimedia_acquire.Client("https://example.org/contact", interval=1.0)
    result, _ = client.get("https://example.org/api.php", max_bytes=100)
    assert result == b"{}"
    assert len(calls) == 2
    assert any(delay >= 2 for delay in slept)
    assert "KurdishNLPResearch" in calls[0][0].get_header("User-agent")


def test_cli_dry_run_and_invalid_language(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "acquire",
                "wikimedia",
                "--languages",
                "ar",
                "--max-pages",
                "5",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "rolling-revision-api" in capsys.readouterr().out
    assert main(["acquire", "wikimedia", "--languages", "sdh", "--dry-run"]) == 1


def test_page_cache_checksum_is_content_sensitive(tmp_path: Path) -> None:
    page = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    _write_api_fixture(tmp_path, "ar", [page])
    manifest = wikimedia.manifests_for(("ar",))["ar"]
    wikimedia_acquire.verify_receipt(manifest, "arwiki", tmp_path)
    cache = tmp_path / "arwiki/pages.jsonl"
    assert hashlib.sha256(cache.read_bytes()).hexdigest()
    cache.write_bytes(cache.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        wikimedia_acquire.verify_receipt(manifest, "arwiki", tmp_path)


def test_missing_and_invalid_cached_metadata(tmp_path: Path) -> None:
    manifest = wikimedia.manifests_for(("ar",))["ar"]
    with pytest.raises(FileNotFoundError):
        wikimedia_acquire.verify_receipt(manifest, "arwiki", tmp_path)
    page = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    with pytest.raises(ValueError, match="ID is invalid"):
        wikimedia_acquire._write_pages(
            tmp_path / "bad.jsonl", [{**page, "revision_id": None}]
        )


def test_revision_safe_document_grouping(tmp_path: Path) -> None:
    page = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    pages = [
        {**page, "wikitext": "هذا نص عربي يتحدث عن تاريخ مدينة قديمة."},
        {
            **page,
            "revision_id": "502",
            "wikitext": "هذا نص عربي آخر يقدم معلومات جديدة عن المنطقة.",
        },
    ]
    _write_api_fixture(tmp_path / "raw", "ar", pages)
    report = wikimedia.build(
        ("ar",), raw_dir=tmp_path / "raw", output_dir=tmp_path / "out"
    )
    assert report["document_counts"]["ar"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/wikimedia.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert len({row["document_id"] for row in rows}) == 1
    assert len({row["split"] for row in rows}) == 1
    capped = wikimedia.build(
        ("ar",),
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "out",
        max_segments_per_language=1,
    )
    assert capped["segments_accepted"] == 1
    assert capped["rejection_counts"]["segment_limit"] == 1


def test_dump_checksum_mismatch_and_resume_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"small official dump bytes"
    target = tmp_path / "source.bz2"
    partial = tmp_path / "source.bz2.part"
    partial.write_bytes(content[:6])
    md5 = hashlib.md5(content).hexdigest()
    sha1 = hashlib.sha1(content).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        wikimedia_acquire._verify_dump(partial, 6, "0" * 32, "0" * 40)

    class Response:
        status = 206
        headers = Message()

        def __init__(self):
            self.headers["Content-Range"] = f"bytes 6-{len(content) - 1}/{len(content)}"
            self.left = content[6:]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            chunk, self.left = self.left, b""
            return chunk

    calls = []

    def fake_open(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(wikimedia_acquire, "urlopen", fake_open)
    client = wikimedia_acquire.Client("https://example.org/contact")
    monkeypatch.setattr(client, "_pace", lambda: None)
    verified = wikimedia_acquire._download_dump(
        client,
        "https://dumps.wikimedia.org/example",
        target,
        expected_size=len(content),
        expected_md5=md5,
        expected_sha1=sha1,
    )
    assert verified["sha256"] == hashlib.sha256(content).hexdigest()
    assert target.read_bytes() == content
    assert calls[0][0].get_header("Range") == "bytes=6-"
    wikimedia_acquire._download_dump(
        client,
        "https://dumps.wikimedia.org/example",
        target,
        expected_size=len(content),
        expected_md5=md5,
        expected_sha1=sha1,
    )
    assert len(calls) == 1


def test_api_error_is_not_silently_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = wikimedia_acquire.Client("https://example.org/contact")
    monkeypatch.setattr(
        client,
        "get",
        lambda *_args, **_kwargs: (b'{"error":{"code":"badrequest"}}', Message()),
    )
    with pytest.raises(ValueError, match="API error"):
        client.api("https://example.org/api.php", {"action": "query"})

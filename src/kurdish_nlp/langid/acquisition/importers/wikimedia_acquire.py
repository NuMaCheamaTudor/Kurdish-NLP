"""Explicit Wikimedia acquisition; imports and tests never invoke this module's I/O."""

from __future__ import annotations

import bz2
import hashlib
import heapq
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from kurdish_nlp.langid.acquisition.manifests import SourceManifest

CONTACT_DEFAULT = "https://github.com/NuMaCheamaTudor/Kurdish-NLP/issues"
MAX_DUMP_BYTES = 500_000_000
MAX_API_BYTES = 16_000_000
RECEIPT_NAME = "receipt.json"
PAGE_CACHE_NAME = "pages.jsonl"
_RETRIABLE = {429, 500, 502, 503, 504}
_CONTENT_RANGE = re.compile(r"bytes (\d+)-\d+/\d+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element if _local(item.tag) == name), None)


def _value(element: ET.Element | None, name: str) -> str | None:
    child = _child(element, name) if element is not None else None
    return child.text if child is not None else None


def iter_dump_pages(archive: Path):
    """Stream current-revision XML; at most one XML page is retained."""
    with bz2.open(archive, "rb") as source:
        parser = ET.iterparse(source, events=("start", "end"))
        _, root = next(parser)
        for event, element in parser:
            if event != "end" or _local(element.tag) != "page":
                continue
            revision = _child(element, "revision")
            contributor = (
                _child(revision, "contributor") if revision is not None else None
            )
            text_node = _child(revision, "text") if revision is not None else None
            page = {
                "page_id": _value(element, "id"),
                "namespace": _value(element, "ns"),
                "title": _value(element, "title"),
                "revision_id": _value(revision, "id"),
                "revision_timestamp": _value(revision, "timestamp"),
                "contributor": _value(contributor, "username"),
                "wikitext": text_node.text or "" if text_node is not None else "",
                "redirect": _child(element, "redirect") is not None,
            }
            yield page
            root.clear()


def _rank(page_id: int, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{page_id}".encode()).digest()[:8], "big"
    )


def _validate_page(page: dict[str, Any]) -> None:
    required = {
        "page_id",
        "namespace",
        "title",
        "revision_id",
        "revision_timestamp",
        "contributor",
        "wikitext",
        "redirect",
    }
    if not isinstance(page, dict) or required - page.keys():
        raise ValueError("cached Wikimedia page is missing required metadata")
    if any(
        not isinstance(page[key], str) or not page[key].isdigit()
        for key in ("page_id", "namespace", "revision_id")
    ):
        raise ValueError("cached Wikimedia page/revision/namespace ID is invalid")
    for key in ("title", "revision_timestamp", "wikitext"):
        if not isinstance(page[key], str):
            raise TypeError(f"cached Wikimedia {key} must be text")
    if not page["title"] or not page["revision_timestamp"]:
        raise ValueError("cached Wikimedia page title/timestamp is missing")
    if page["contributor"] is not None and not isinstance(page["contributor"], str):
        raise TypeError("cached Wikimedia contributor must be text or null")
    if type(page["redirect"]) is not bool:
        raise TypeError("cached Wikimedia redirect flag must be boolean")


def select_dump_page_ids(
    archive: Path, *, seed: int, max_pages: int
) -> tuple[list[int], int]:
    """Seeded, order-independent bottom-k sample from every mainspace page ID."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    selected: list[tuple[int, int]] = []
    examined = 0
    for page in iter_dump_pages(archive):
        if page["namespace"] != "0" or not str(page["page_id"]).isdigit():
            continue
        examined += 1
        page_id = int(page["page_id"])
        score = _rank(page_id, seed)
        item = (-score, -page_id)
        if len(selected) < max_pages:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    return sorted(-item[1] for item in selected), examined


class Client:
    """Sequential, identified client with maxlag, throttling and bounded retries."""

    def __init__(self, contact: str, *, interval: float = 1.1, timeout: float = 30.0):
        if not contact or not (contact.startswith("https://") or "@" in contact):
            raise ValueError("Wikimedia contact must be an email or https URL")
        if interval < 1.0:
            raise ValueError("Wikimedia request interval must be at least one second")
        self.user_agent = f"KurdishNLPResearch/0.1 ({contact}; research acquisition)"
        self.interval = interval
        self.timeout = timeout
        self._last_request = 0.0

    def _pace(self) -> None:
        remaining = self.interval - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    def get(
        self, url: str, *, max_bytes: int, headers: dict[str, str] | None = None
    ) -> tuple[bytes, Any]:
        for attempt in range(5):
            self._pace()
            request = Request(
                url,
                headers={"User-Agent": self.user_agent, **(headers or {})},
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(max_bytes + 1)
                    if len(raw) > max_bytes:
                        raise ValueError(
                            f"Wikimedia response exceeds {max_bytes} bytes"
                        )
                    return raw, response.headers
            except HTTPError as error:
                if error.code not in _RETRIABLE or attempt == 4:
                    raise
                delay = min(60.0, 2.0**attempt)
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    if retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                    else:
                        try:
                            date = parsedate_to_datetime(retry_after)
                            delay = max(
                                delay, (date - datetime.now(UTC)).total_seconds()
                            )
                        except (TypeError, ValueError):
                            pass
                time.sleep(max(0.0, delay))
            except (TimeoutError, URLError):
                if attempt == 4:
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        raise AssertionError("unreachable retry loop")

    def api(self, url: str, params: dict[str, str | int]) -> dict[str, Any]:
        query = {
            "format": "json",
            "formatversion": 2,
            "maxlag": 1,
            **params,
        }
        endpoint = url + "?" + urlencode(query)
        for attempt in range(5):
            raw, _ = self.get(endpoint, max_bytes=MAX_API_BYTES)
            result = json.loads(raw)
            if "error" not in result:
                return result
            if result["error"].get("code") != "maxlag" or attempt == 4:
                raise ValueError(f"Wikimedia API error: {result['error']}")
            time.sleep(min(60.0, 2.0**attempt))
        raise AssertionError("unreachable maxlag loop")


def _download_dump(
    client: Client,
    url: str,
    target: Path,
    *,
    expected_size: int,
    expected_md5: str,
    expected_sha1: str,
) -> dict[str, Any]:
    if expected_size > MAX_DUMP_BYTES:
        raise ValueError("dump exceeds configured safe download bound")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == expected_size:
            return _verify_dump(target, expected_size, expected_md5, expected_sha1)
        raise ValueError("existing dump has incorrect size; refusing to overwrite")
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("partial dump exceeds official size")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    client._pace()
    request = Request(url, headers={"User-Agent": client.user_agent, **headers})
    with urlopen(request, timeout=client.timeout) as response:
        if offset:
            match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or int(match.group(1)) != offset:
                raise ValueError("dump server did not honor resume offset")
        elif response.status != 200:
            raise ValueError(f"unexpected dump response {response.status}")
        with partial.open("ab" if offset else "wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                if output.tell() > expected_size:
                    raise ValueError("dump exceeded official size")
    metadata = _verify_dump(partial, expected_size, expected_md5, expected_sha1)
    os.replace(partial, target)
    return metadata


def _verify_dump(path: Path, size: int, md5: str, sha1: str) -> dict[str, Any]:
    if path.stat().st_size != size:
        raise ValueError("Wikimedia dump size mismatch")
    hashes = {name: hashlib.new(name) for name in ("md5", "sha1", "sha256")}
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            for digest in hashes.values():
                digest.update(block)
    if hashes["md5"].hexdigest() != md5 or hashes["sha1"].hexdigest() != sha1:
        raise ValueError("Wikimedia official dump checksum mismatch")
    return {
        "bytes": size,
        "md5": md5,
        "sha1": sha1,
        "sha256": hashes["sha256"].hexdigest(),
    }


def _page_from_api(value: dict[str, Any]) -> dict[str, Any]:
    revision = value.get("revisions", [{}])[0]
    content = revision.get("slots", {}).get("main", {}).get("content")
    if not isinstance(content, str):
        raise TypeError("API revision lacks main-slot wikitext")
    return {
        "page_id": str(value["pageid"]),
        "namespace": str(value["ns"]),
        "title": value["title"],
        "revision_id": str(revision["revid"]),
        "revision_timestamp": revision["timestamp"],
        "contributor": revision.get("user"),
        "wikitext": content,
        "redirect": content.lstrip().lower().startswith("#redirect"),
    }


def _write_pages(path: Path, pages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for page in pages:
        _validate_page(page)
    temporary = path.with_suffix(".jsonl.part")
    with temporary.open("wb") as output:
        for page in sorted(pages, key=lambda item: int(item["page_id"])):
            output.write(_json_bytes(page))
    os.replace(temporary, path)


def acquire_dump(
    manifest: SourceManifest,
    project: str,
    raw_dir: Path,
    *,
    max_pages: int,
    seed: int,
    client: Client,
) -> dict[str, Any]:
    directory = raw_dir / project
    directory.mkdir(parents=True, exist_ok=True)
    date = manifest.source_version
    filename = f"{project}-{date}-pages-articles-multistream.xml.bz2"
    status_url = f"https://dumps.wikimedia.org/{project}/{date}/dumpstatus.json"
    status_path = directory / "dumpstatus.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        raw, _ = client.get(status_url, max_bytes=5_000_000)
        status = json.loads(raw)
        _atomic_json(status_path, status)
    job = status.get("jobs", {}).get("articlesmultistreamdump", {})
    if job.get("status") != "done" or filename not in job.get("files", {}):
        raise ValueError(f"official {project} article dump is not complete")
    info = job["files"][filename]
    url = f"https://dumps.wikimedia.org/{project}/{date}/{filename}"
    archive = directory / filename
    source = _download_dump(
        client,
        url,
        archive,
        expected_size=int(info["size"]),
        expected_md5=info["md5"],
        expected_sha1=info["sha1"],
    )
    selected, frame_size = select_dump_page_ids(archive, seed=seed, max_pages=max_pages)
    chosen = set(selected)
    pages = [
        page
        for page in iter_dump_pages(archive)
        if page["namespace"] == "0"
        and str(page["page_id"]).isdigit()
        and int(page["page_id"]) in chosen
    ]
    _write_pages(directory / PAGE_CACHE_NAME, pages)
    return _receipt(
        manifest,
        project,
        directory,
        method="dated-pages-articles-dump",
        pages=pages,
        source={"url": url, **source, "dumpstatus_sha256": sha256_file(status_path)},
        sampling={
            "method": "seeded-bottom-k-mainspace-page-id",
            "seed": seed,
            "frame_size": frame_size,
            "target_pages": max_pages,
        },
    )


def acquire_api(
    manifest: SourceManifest,
    project: str,
    raw_dir: Path,
    *,
    max_pages: int,
    seed: int,
    client: Client,
) -> dict[str, Any]:
    if max_pages > 500:
        raise ValueError("API acquisition is bounded to at most 500 pages per edition")
    directory = raw_dir / project
    pages_dir = directory / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    frame_path = directory / "frame.json"
    if frame_path.exists():
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        if frame.get("seed") != seed:
            raise ValueError("existing API frame uses another seed")
    else:
        frame = {"seed": seed, "page_ids": [], "sample_response_sha256": []}
    by_id = {int(item["page_id"]): item for item in frame["page_ids"]}
    while len(by_id) < max_pages:
        request_count = min(500, max(20, max_pages - len(by_id)))
        result = client.api(
            manifest.source_url,
            {
                "action": "query",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": request_count,
                "prop": "info",
            },
        )
        raw = _json_bytes(result)
        frame["sample_response_sha256"].append(hashlib.sha256(raw).hexdigest())
        for page in result.get("query", {}).get("pages", []):
            if page.get("ns") == 0 and page.get("lastrevid"):
                by_id.setdefault(
                    int(page["pageid"]),
                    {
                        "page_id": int(page["pageid"]),
                        "revision_id": int(page["lastrevid"]),
                        "title_at_selection": page["title"],
                    },
                )
        frame["page_ids"] = list(by_id.values())
        _atomic_json(frame_path, frame)
        if not result.get("query", {}).get("pages"):
            raise ValueError("API random-page sampling returned no mainspace pages")
    selected = frame["page_ids"][:max_pages]
    missing = [
        item
        for item in selected
        if not (pages_dir / f"{item['page_id']}.json").exists()
    ]
    for start in range(0, len(missing), 5):
        batch = missing[start : start + 5]
        result = client.api(
            manifest.source_url,
            {
                "action": "query",
                "prop": "revisions",
                "revids": "|".join(str(item["revision_id"]) for item in batch),
                "rvprop": "ids|timestamp|user|sha1|content",
                "rvslots": "main",
            },
        )
        returned = {}
        for item in result.get("query", {}).get("pages", []):
            if item.get("revisions"):
                returned[int(item["pageid"])] = _page_from_api(item)
        for item in batch:
            page = returned.get(item["page_id"])
            if page is None or int(page["revision_id"]) != item["revision_id"]:
                raise ValueError(
                    f"pinned revision unavailable for page {item['page_id']}"
                )
            _atomic_json(pages_dir / f"{item['page_id']}.json", page)
    pages = [
        json.loads((pages_dir / f"{item['page_id']}.json").read_text(encoding="utf-8"))
        for item in selected
    ]
    _write_pages(directory / PAGE_CACHE_NAME, pages)
    return _receipt(
        manifest,
        project,
        directory,
        method="revision-pinned-action-api",
        pages=pages,
        source={
            "url": manifest.source_url,
            "sample_frame_sha256": sha256_file(frame_path),
            "cached_page_bytes": sum(
                (pages_dir / f"{item['page_id']}.json").stat().st_size
                for item in selected
            ),
            "network_transfer_bytes": None,
        },
        sampling={
            "method": "server-random-frozen-page-and-revision-ids",
            "seed": seed,
            "frame_size": len(by_id),
            "target_pages": max_pages,
        },
    )


def _receipt(
    manifest: SourceManifest,
    project: str,
    directory: Path,
    *,
    method: str,
    pages: list[dict[str, Any]],
    source: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    page_metadata = [
        {
            "page_id": page["page_id"],
            "revision_id": page["revision_id"],
            "content_sha256": hashlib.sha256(
                page["wikitext"].encode("utf-8")
            ).hexdigest(),
        }
        for page in sorted(pages, key=lambda item: int(item["page_id"]))
    ]
    stable = {
        "manifest_version": manifest.manifest_version,
        "manifest_sha256": hashlib.sha256(_json_bytes(manifest.to_dict())).hexdigest(),
        "method": method,
        "project": project,
        "pages": page_metadata,
        "pages_sha256": sha256_file(directory / PAGE_CACHE_NAME),
        "source": source,
        "sampling": sampling,
    }
    snapshot_id = hashlib.sha256(_json_bytes(stable)).hexdigest()
    receipt = {
        "schema_version": 1,
        "source_id": manifest.source_id,
        "acquired_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_id,
        **stable,
    }
    _atomic_json(directory / RECEIPT_NAME, receipt)
    return receipt


def verify_receipt(
    manifest: SourceManifest, project: str, raw_dir: Path
) -> dict[str, Any]:
    directory = raw_dir / project
    receipt = json.loads((directory / RECEIPT_NAME).read_text(encoding="utf-8"))
    if (
        receipt["source_id"] != manifest.source_id
        or receipt["manifest_version"] != manifest.manifest_version
        or receipt.get("manifest_sha256")
        != hashlib.sha256(_json_bytes(manifest.to_dict())).hexdigest()
    ):
        raise ValueError("Wikimedia receipt does not match source manifest")
    stable = {
        key: receipt[key]
        for key in (
            "manifest_version",
            "manifest_sha256",
            "method",
            "project",
            "pages",
            "pages_sha256",
            "source",
            "sampling",
        )
    }
    if (
        receipt["project"] != project
        or hashlib.sha256(_json_bytes(stable)).hexdigest() != receipt["snapshot_id"]
    ):
        raise ValueError("Wikimedia receipt snapshot identity mismatch")
    if sha256_file(directory / PAGE_CACHE_NAME) != receipt["pages_sha256"]:
        raise ValueError("Wikimedia cached page checksum mismatch")
    if receipt["method"] == "dated-pages-articles-dump":
        archive = directory / Path(receipt["source"]["url"]).name
        source = receipt["source"]
        if sha256_file(directory / "dumpstatus.json") != source["dumpstatus_sha256"]:
            raise ValueError("Wikimedia official dumpstatus checksum mismatch")
        verified = _verify_dump(archive, source["bytes"], source["md5"], source["sha1"])
        if verified["sha256"] != source["sha256"]:
            raise ValueError("Wikimedia dump SHA-256 mismatch")
    with (directory / PAGE_CACHE_NAME).open(encoding="utf-8") as cache:
        seen = []
        for line in cache:
            page = json.loads(line)
            _validate_page(page)
            seen.append(
                {
                    "page_id": page["page_id"],
                    "revision_id": page["revision_id"],
                    "content_sha256": hashlib.sha256(
                        page["wikitext"].encode("utf-8")
                    ).hexdigest(),
                }
            )
    if seen != receipt["pages"]:
        raise ValueError("Wikimedia page/revision/content receipt mismatch")
    return receipt


def disk_preflight(raw_dir: Path, estimated_bytes: int) -> dict[str, int]:
    existing = (
        raw_dir
        if raw_dir.exists()
        else next(parent for parent in raw_dir.parents if parent.exists())
    )
    usage = shutil.disk_usage(existing)
    if usage.free < estimated_bytes * 3:
        raise ValueError("insufficient free disk for Wikimedia acquisition and cache")
    return {
        "free_bytes": usage.free,
        "estimated_download_bytes": estimated_bytes,
        "estimated_working_bytes": estimated_bytes * 3,
    }

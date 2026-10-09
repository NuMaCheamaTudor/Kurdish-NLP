"""Pinned Universal Dependencies acquisition and text-only external benchmark.

Only ``acquire`` makes network requests. ``build`` reads verified local archives
and existing datasets; it never changes training records or source files.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import tarfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from kurdish_nlp.langid.acquisition.benchmark import (
    BENCHMARK,
    PARTITION,
    ExternalBenchmarkRecord,
    validate_benchmark,
)
from kurdish_nlp.langid.acquisition.licenses import LicenseId
from kurdish_nlp.langid.acquisition.manifests import SourceManifest, load_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.normalization import normalize_text

VERSION = "v2.18"
DEFAULT_RAW_DIR = Path("data/langid/raw/ud/v2.18")
DEFAULT_OUTPUT_DIR = Path("data/langid/benchmarks/ud/v2.18")
CANONICAL_DIR = Path("data/langid/processed")
LEGACY_DIR = Path("data/raw")
MANIFEST_DIR = Path("configs/langid/sources")
TREEBANKS = {
    "sdh": ("sdh_garrusi", "UD_Southern_Kurdish-Garrusi", ("train", "dev", "test")),
    "kmr": ("kmr_kurmanji", "UD_Northern_Kurdish-Kurmanji", ("train", "test")),
}
CONLLU_COLUMNS = 10
TOKEN_ID = re.compile(r"[1-9][0-9]*\Z")
MWT_ID = re.compile(r"([1-9][0-9]*)-([1-9][0-9]*)\Z")
EMPTY_ID = re.compile(r"[1-9][0-9]*\.[1-9][0-9]*\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_languages(languages: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(languages or TREEBANKS))
    if not selected or set(selected) - TREEBANKS.keys():
        raise ValueError(
            f"unsupported UD languages: {sorted(set(selected) - TREEBANKS.keys())}"
        )
    return tuple(sorted(selected))


def manifests_for(languages: Iterable[str]) -> dict[str, SourceManifest]:
    return {
        language: load_manifest(
            MANIFEST_DIR
            / f"ud-{language}-{TREEBANKS[language][0].split('_')[1]}-v2.18.v1.json"
        )
        for language in selected_languages(languages)
    }


def _check_manifest(manifest: SourceManifest, language: str) -> None:
    treebank, repository, _ = TREEBANKS[language]
    if manifest.source_id != f"ud-{language}-{treebank.split('_')[1]}-v2-18":
        raise ValueError("UD source ID does not match its treebank")
    if (
        manifest.source_url
        != f"https://github.com/UniversalDependencies/{repository}/tree/r2.18"
    ):
        raise ValueError("UD source URL must be the official r2.18 tag")
    if manifest.source_version != "r2.18" or manifest.checksum_sha256 is None:
        raise ValueError("UD archive must have a pinned r2.18 checksum")
    if manifest.license.identifier is not LicenseId.CC_BY_SA_4:
        raise ValueError("UD treebank license must be reviewed CC-BY-SA-4.0")
    if set(manifest.allowed_roles) != {"external_evaluation", "benchmark"}:
        raise ValueError("UD manifests must permit evaluation-related roles only")


def archive_url(language: str) -> str:
    return f"https://codeload.github.com/UniversalDependencies/{TREEBANKS[language][1]}/tar.gz/refs/tags/r2.18"


def archive_path(language: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    return raw_dir / f"{TREEBANKS[language][0]}-r2.18.tar.gz"


def plan_acquisition(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
) -> dict[str, Any]:
    manifests = manifests_for(selected_languages(languages))
    sources: dict[str, Any] = {}
    for language, manifest in manifests.items():
        _check_manifest(manifest, language)
        decision = evaluate(manifest, role="external_evaluation", profile=profile)
        sources[language] = {
            "treebank": TREEBANKS[language][0],
            "version": VERSION,
            "url": archive_url(language),
            "destination": str(archive_path(language, raw_dir)),
            "expected_sha256": manifest.checksum_sha256,
            "policy": decision.to_dict(),
        }
    return {"sources": sources, "network_required": False, "evaluation_only": True}


def acquire(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    profile: str = "commercial",
    opener: Any = None,
) -> dict[str, dict[str, Any]]:
    """Explicit network operation; bad cached/downloaded archives fail closed."""
    plan = plan_acquisition(languages, raw_dir=raw_dir, profile=profile)
    if any(not source["policy"]["allowed"] for source in plan["sources"].values()):
        raise ValueError("UD external evaluation blocked by source policy")
    results = {}
    for language, source in plan["sources"].items():
        destination = Path(source["destination"])
        expected = source["expected_sha256"]
        if destination.exists():
            actual = sha256_file(destination)
            if actual != expected:
                raise ValueError(
                    f"cached UD {language} checksum mismatch: {actual} != {expected}"
                )
            reused = True
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".part")
            request = Request(
                source["url"], headers={"User-Agent": "kurdish-nlp-ud-benchmark/1"}
            )
            with (
                (opener or urlopen)(request, timeout=60) as response,
                partial.open("wb") as output,
            ):
                if response.status != 200:
                    raise ValueError(f"unexpected UD HTTP status: {response.status}")
                shutil.copyfileobj(response, output)
            actual = sha256_file(partial)
            if actual != expected:
                raise ValueError(
                    f"downloaded UD {language} checksum mismatch: {actual} != {expected}"
                )
            os.replace(partial, destination)
            reused = False
        results[language] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": expected,
            "reused": reused,
        }
    return results


@dataclass(frozen=True, slots=True)
class Sentence:
    line_number: int
    ordinal: int
    sent_id: str | None
    text: str
    original_text: str | None
    document_id: str | None
    paragraph_id: str | None
    comments: tuple[str, ...]
    token_count: int
    multiword_count: int
    empty_node_count: int
    reconstruction: str
    warnings: tuple[str, ...]
    annotation_sha256: str


def _parse_sentence(
    lines: list[str],
    *,
    line_number: int,
    ordinal: int,
    document_id: str | None,
    paragraph_id: str | None,
) -> Sentence:
    comments: list[str] = []
    metadata: dict[str, str] = {}
    tokens: list[tuple[str, str, str]] = []
    multiwords: dict[int, tuple[int, str, str]] = {}
    ordinary_ids: set[int] = set()
    empty_count = 0
    for line in lines:
        if line.startswith("#"):
            comments.append(line)
            if " = " in line:
                key, value = line[2:].split(" = ", 1)
                if key in metadata:
                    raise ValueError(f"duplicate {key} comment")
                metadata[key] = value
            continue
        columns = line.split("\t")
        if len(columns) != CONLLU_COLUMNS:
            raise ValueError(f"CoNLL-U token has {len(columns)} rather than 10 columns")
        token_id, form, misc = columns[0], columns[1], columns[9]
        if not form:
            raise ValueError("missing token FORM")
        mwt = MWT_ID.fullmatch(token_id)
        if mwt:
            start, end = map(int, mwt.groups())
            if end <= start or start in multiwords:
                raise ValueError("invalid or duplicate multiword token range")
            multiwords[start] = (end, form, misc)
        elif TOKEN_ID.fullmatch(token_id):
            number = int(token_id)
            if number in ordinary_ids:
                raise ValueError("duplicate integer token ID")
            ordinary_ids.add(number)
        elif EMPTY_ID.fullmatch(token_id):
            empty_count += 1
        else:
            raise ValueError(f"invalid CoNLL-U token ID: {token_id}")
        tokens.append((token_id, form, misc))
    if not ordinary_ids:
        raise ValueError("sentence has no syntactic tokens")
    if ordinary_ids != set(range(1, max(ordinary_ids) + 1)):
        raise ValueError("non-consecutive syntactic token IDs")
    for start, (end, _, _) in multiwords.items():
        if not set(range(start, end + 1)) <= ordinary_ids:
            raise ValueError("multiword token range lacks syntactic words")
    surfaces: list[tuple[str, str]] = []
    skip_through = 0
    ordinary_misc = {
        int(token_id): misc
        for token_id, _, misc in tokens
        if TOKEN_ID.fullmatch(token_id)
    }
    for token_id, form, misc in tokens:
        if EMPTY_ID.fullmatch(token_id):
            continue
        mwt = MWT_ID.fullmatch(token_id)
        if mwt:
            start, end = map(int, mwt.groups())
            if start <= skip_through:
                raise ValueError("overlapping multiword token ranges")
            spacing = misc if misc != "_" else ordinary_misc[end]
            surfaces.append((form, spacing))
            skip_through = end
        elif TOKEN_ID.fullmatch(token_id) and int(token_id) > skip_through:
            surfaces.append((form, misc))
    reconstructed = "".join(
        form + ("" if "SpaceAfter=No" in misc.split("|") else " ")
        for form, misc in surfaces
    ).rstrip()
    original = metadata.get("text")
    if original and normalize_text(original):
        text, reconstruction = original, "official_text"
    else:
        if any(form == "_" for form, _ in surfaces):
            raise ValueError("cannot reconstruct text from missing token FORM")
        text, reconstruction = reconstructed, "reconstructed"
    if not normalize_text(text):
        raise ValueError("empty sentence text")
    warnings = []
    if not metadata.get("sent_id"):
        warnings.append("missing_sent_id")
    if reconstruction == "reconstructed":
        warnings.append("missing_or_empty_text")
    if any(form == "_" for _, form, _ in tokens):
        warnings.append("missing_token_form_in_annotations")
    if original is not None and normalize_text(original) != normalize_text(
        reconstructed
    ):
        warnings.append("text_token_reconstruction_disagrees")
    return Sentence(
        line_number,
        ordinal,
        metadata.get("sent_id"),
        text,
        original,
        document_id,
        paragraph_id,
        tuple(comments),
        len(surfaces),
        len(multiwords),
        empty_count,
        reconstruction,
        tuple(warnings),
        text_hash("\n".join(lines) + "\n"),
    )


def parse_conllu(
    source: Iterable[str],
) -> Iterator[tuple[Sentence | None, dict[str, Any] | None]]:
    """Stream sentences and explicit parse errors, carrying only actual document metadata."""
    block: list[str] = []
    first_line = 1
    document_id: str | None = None
    paragraph_id: str | None = None
    ordinal = 0

    def finish(
        lines: list[str], line: int
    ) -> tuple[Sentence | None, dict[str, Any] | None]:
        nonlocal document_id, paragraph_id, ordinal
        ordinal += 1
        for comment in lines:
            if comment.startswith("# newdoc"):
                document_id = comment.partition(" = ")[2] or None
                paragraph_id = None
            elif comment.startswith("# newpar"):
                paragraph_id = comment.partition(" = ")[2] or None
        try:
            return _parse_sentence(
                lines,
                line_number=line,
                ordinal=ordinal,
                document_id=document_id,
                paragraph_id=paragraph_id,
            ), None
        except ValueError as error:
            return None, {"ordinal": ordinal, "line_number": line, "reason": str(error)}

    for number, raw_line in enumerate(source, 1):
        line = raw_line.rstrip("\r\n")
        if line:
            if not block:
                first_line = number
            block.append(line)
        elif block:
            yield finish(block, first_line)
            block = []
    if block:
        yield finish(block, first_line)


def _script(text: str) -> dict[str, Any]:
    latin = arabic = other = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            latin += 1
        elif "ARABIC" in name:
            arabic += 1
        else:
            other += 1
    label = (
        "mixed"
        if latin and arabic
        else "latin"
        if latin
        else "arabic"
        if arabic
        else "other"
    )
    return {
        "label": label,
        "latin_letters": latin,
        "arabic_letters": arabic,
        "other_letters": other,
    }


def _member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [
        item
        for item in archive.getmembers()
        if item.isfile() and item.name.endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {suffix} in UD archive")
    return matches[0]


def _read_sources(
    languages: tuple[str, ...], raw_dir: Path, profile: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    manifests = manifests_for(languages)
    for language in languages:
        manifest = manifests[language]
        _check_manifest(manifest, language)
        if not evaluate(manifest, role="external_evaluation", profile=profile).allowed:
            raise ValueError(f"UD {language} external evaluation blocked by policy")
        path = archive_path(language, raw_dir)
        if not path.is_file():
            raise FileNotFoundError(
                f"UD {language} archive missing: {path}; run acquire ud"
            )
        actual = sha256_file(path)
        if actual != manifest.checksum_sha256:
            raise ValueError(
                f"UD {language} archive checksum mismatch: {actual} != {manifest.checksum_sha256}"
            )
        treebank, _, splits = TREEBANKS[language]
        stats[language] = {
            "archive": str(path),
            "archive_sha256": actual,
            "archive_bytes": path.stat().st_size,
            "source_sentences": 0,
            "accepted": 0,
            "source_tokens": 0,
            "multiword_tokens": 0,
            "empty_nodes": 0,
            "missing_text": 0,
            "missing_sent_id": 0,
            "document_id_present": 0,
            "paragraph_id_present": 0,
            "reconstructed": 0,
            "text_reconstruction_disagreements": 0,
            "malformed": [],
            "splits": {},
            "source_files": {},
        }
        with tarfile.open(path, "r:gz") as archive:
            for split in splits:
                filename = f"{treebank}-ud-{split}.conllu"
                member = _member(archive, filename)
                binary = archive.extractfile(member)
                if binary is None:
                    raise ValueError(f"unreadable archive member: {filename}")
                payload = binary.read()
                member_sha = hashlib.sha256(payload).hexdigest()
                stats[language]["source_files"][filename] = {
                    "sha256": member_sha,
                    "bytes": len(payload),
                }
                count = 0
                for sentence, problem in parse_conllu(
                    io.StringIO(payload.decode("utf-8-sig"))
                ):
                    count += 1
                    stats[language]["source_sentences"] += 1
                    if problem:
                        stats[language]["malformed"].append(
                            {"source_file": filename, **problem}
                        )
                        continue
                    assert sentence is not None
                    candidates.append(
                        {
                            "language": language,
                            "treebank": treebank,
                            "split": split,
                            "source_file": filename,
                            "source_file_sha256": member_sha,
                            "sentence": sentence,
                            "source_url": manifest.source_url,
                            "source_id": manifest.source_id,
                            "manifest_version": manifest.manifest_version,
                            "archive_sha256": actual,
                        }
                    )
                    stats[language]["accepted"] += 1
                    stats[language]["source_tokens"] += sentence.token_count
                    stats[language]["multiword_tokens"] += sentence.multiword_count
                    stats[language]["empty_nodes"] += sentence.empty_node_count
                    stats[language]["missing_text"] += sentence.original_text is None
                    stats[language]["missing_sent_id"] += sentence.sent_id is None
                    stats[language]["document_id_present"] += (
                        sentence.document_id is not None
                    )
                    stats[language]["paragraph_id_present"] += (
                        sentence.paragraph_id is not None
                    )
                    stats[language]["reconstructed"] += (
                        sentence.reconstruction == "reconstructed"
                    )
                    stats[language]["text_reconstruction_disagreements"] += (
                        "text_token_reconstruction_disagrees" in sentence.warnings
                    )
                stats[language]["splits"][split] = count
    return candidates, stats


def _shingles(text: str) -> set[str]:
    words = normalize_text(text).casefold().split()
    return {" ".join(words[index : index + 5]) for index in range(len(words) - 4)}


def _near_ratio(left: str, right: str) -> float:
    if abs(len(left) - len(right)) > max(len(left), len(right)) * 0.25:
        return 0.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _cross_source(
    candidates: list[dict[str, Any]], canonical_dir: Path
) -> dict[str, Any]:
    exact: dict[str, list[int]] = defaultdict(list)
    normalized: dict[str, list[int]] = defaultdict(list)
    shingles: dict[str, set[int]] = defaultdict(set)
    overlap: dict[int, list[dict[str, str]]] = defaultdict(list)
    for index, item in enumerate(candidates):
        text = item["sentence"].text
        exact[text].append(index)
        normalized[normalize_text(text)].append(index)
        for shingle in _shingles(text):
            shingles[shingle].add(index)
    report: dict[str, Any] = {}
    for source_name, filename in (
        ("parme", "parme.sdh.jsonl"),
        ("tatoeba", "tatoeba.jsonl"),
        ("wikimedia", "wikimedia.jsonl"),
    ):
        path = canonical_dir / filename
        if not path.is_file():
            report[source_name] = {"available": False, "rows_scanned": 0, "matches": []}
            continue
        rows = 0
        matches: list[dict[str, str]] = []
        sdh_scripts: Counter[str] = Counter()
        for raw_line in path.open(encoding="utf-8"):
            row = json.loads(raw_line)
            rows += 1
            text = row["text"]
            if row["label"] == "sdh":
                sdh_scripts[_script(text)["label"]] += 1
            exact_ids = set(exact.get(text, ()))
            normalized_ids = set(normalized.get(normalize_text(text), ()))
            near_ids: set[int] = set()
            for shingle in _shingles(text):
                near_ids.update(shingles.get(shingle, ()))
            for index in sorted(exact_ids | normalized_ids | near_ids):
                ud_text = candidates[index]["sentence"].text
                kind = (
                    "exact"
                    if index in exact_ids
                    else "normalized"
                    if index in normalized_ids
                    else "near"
                )
                if (
                    kind == "near"
                    and _near_ratio(
                        normalize_text(text).casefold(),
                        normalize_text(ud_text).casefold(),
                    )
                    < 0.88
                ):
                    continue
                match = {
                    "candidate_index": index,
                    "source_record_id": row["id"],
                    "source_label": row["label"],
                    "source_split": row.get("split"),
                    "source_document_id": row.get("document_id"),
                    "match_type": kind,
                }
                matches.append(match)
                overlap[index].append({"source": source_name, **match})
        report[source_name] = {
            "available": True,
            "rows_scanned": rows,
            "matches": matches,
            "affected_ud_sentences": len(
                {match["candidate_index"] for match in matches}
            ),
            "label_conflicts": sum(
                candidates[match["candidate_index"]]["language"]
                != match["source_label"]
                for match in matches
            ),
            "sdh_script_distribution": dict(sorted(sdh_scripts.items())),
            "match_types": dict(
                sorted(Counter(match["match_type"] for match in matches).items())
            ),
            "matched_source_splits": dict(
                sorted(
                    Counter(
                        match["source_split"] or "unknown" for match in matches
                    ).items()
                )
            ),
        }
    return {
        "sources": report,
        "by_candidate": overlap,
        "near_method": "All canonical rows streamed; shared normalized 5-word shingles generate candidates, then casefolded SequenceMatcher >=0.88 with <=25% length difference. Near matches shorter than five words or without a common 5-gram are not detected.",
    }


def _legacy_overlap(
    candidates: list[dict[str, Any]], stats: dict[str, Any], legacy_dir: Path
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    affected: set[int] = set()
    for split in TREEBANKS["kmr"][2]:
        filename = f"kmr_kurmanji-ud-{split}.conllu"
        path = legacy_dir / filename
        if not path.is_file() or "kmr" not in stats:
            files[filename] = {
                "available": path.is_file(),
                "byte_identical": None,
                "affected": 0,
            }
            continue
        actual = sha256_file(path)
        official = stats["kmr"]["source_files"][filename]["sha256"]
        identical = actual == official
        indices = (
            {
                index
                for index, item in enumerate(candidates)
                if item["source_file"] == filename
            }
            if identical
            else set()
        )
        if not identical:
            texts = {
                sentence.text
                for sentence, problem in parse_conllu(path.open(encoding="utf-8"))
                if sentence and not problem
            }
            indices = {
                index
                for index, item in enumerate(candidates)
                if item["source_file"] == filename and item["sentence"].text in texts
            }
        affected |= indices
        files[filename] = {
            "available": True,
            "sha256": actual,
            "official_sha256": official,
            "byte_identical": identical,
            "affected": len(indices),
        }
    return {"files": files, "affected_candidate_indices": affected}


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for value in values:
            output.write(
                json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def build(
    languages: Iterable[str] | None = None,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    canonical_dir: Path = CANONICAL_DIR,
    legacy_dir: Path = LEGACY_DIR,
    profile: str = "commercial",
) -> dict[str, Any]:
    """Offline deterministic import, contamination audit and linked export."""
    selected = selected_languages(languages)
    candidates, stats = _read_sources(selected, raw_dir, profile)
    cross = _cross_source(candidates, canonical_dir)
    legacy = _legacy_overlap(candidates, stats, legacy_dir)
    comparisons_complete = all(
        source["available"] for source in cross["sources"].values()
    )
    legacy_comparison_complete = (
        all(item["available"] for item in legacy["files"].values())
        if "kmr" in selected
        else True
    )
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    sent_ids: Counter[tuple[str, str, str]] = Counter()
    for index, item in enumerate(candidates):
        normalized_groups[normalize_text(item["sentence"].text)].append(index)
        if item["sentence"].sent_id:
            sent_ids[(item["treebank"], item["split"], item["sentence"].sent_id)] += 1
    duplicate_indices = {
        index
        for group in normalized_groups.values()
        if len(group) > 1
        for index in group
    }
    records: list[ExternalBenchmarkRecord] = []
    provenance: list[dict[str, Any]] = []
    script_counts: Counter[str] = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    exclusion_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    for index, item in enumerate(candidates):
        sentence: Sentence = item["sentence"]
        language = item["language"]
        script = _script(sentence.text)
        script_counts[f"{language}:{script['label']}"] += 1
        lengths[language].append(len(sentence.text))
        genre = (
            "grammar-examples"
            if language == "sdh"
            else "wiki"
            if (sentence.sent_id or "").startswith("wiki:")
            else "fiction"
        )
        genre_counts[f"{language}:{genre}"] += 1
        duplicate_id = (
            sent_ids[(item["treebank"], item["split"], sentence.sent_id)] > 1
            if sentence.sent_id
            else False
        )
        identity = (
            sentence.sent_id
            if sentence.sent_id and not duplicate_id
            else f"ordinal:{sentence.ordinal}"
        )
        record_id = f"ud:{item['treebank']}:{item['split']}:{text_hash(identity + chr(0) + sentence.text)[:20]}"
        reasons = []
        if index in duplicate_indices:
            reasons.append("within_ud_duplicate_text")
        if duplicate_id:
            reasons.append("duplicate_source_sent_id")
        if index in legacy["affected_candidate_indices"]:
            reasons.append("legacy_ud_prior_repository_exposure")
        if not comparisons_complete:
            reasons.append("canonical_comparison_unavailable")
        if language == "kmr" and not legacy_comparison_complete:
            reasons.append("legacy_ud_comparison_unavailable")
        matches = cross["by_candidate"].get(index, [])
        if matches:
            reasons.append("canonical_source_overlap")
        for reason in reasons:
            exclusion_counts[f"{language}:{reason}"] += 1
        slice_name = (
            "sdh_garrusi_original_orthography"
            if language == "sdh"
            else f"kmr_kurmanji_{genre}"
        )
        record = ExternalBenchmarkRecord(
            record_id,
            sentence.text,
            language,
            BENCHMARK,
            item["treebank"],
            item["split"],
            PARTITION,
            "universal-dependencies",
            "CC-BY-SA-4.0",
            slice_name,
            not reasons,
            tuple(reasons),
        )
        records.append(record)
        fallback_component = (
            ":".join(sentence.sent_id.split(":")[:2])
            if language == "kmr" and sentence.sent_id and ":" in sentence.sent_id
            else "unknown_document"
        )
        provenance.append(
            {
                "schema_version": 1,
                "evaluation_record_id": record_id,
                "source_id": item["source_id"],
                "source_url": item["source_url"],
                "acquisition_manifest_version": item["manifest_version"],
                "ud_release": VERSION,
                "treebank": item["treebank"],
                "original_split": item["split"],
                "original_sent_id": sentence.sent_id,
                "original_document_id": sentence.document_id,
                "original_paragraph_id": sentence.paragraph_id,
                "fallback_group_id": f"{item['treebank']}:{item['split']}:{fallback_component}",
                "fallback_group_is_document_id": False,
                "source_file": item["source_file"],
                "source_line_number": sentence.line_number,
                "source_sentence_ordinal": sentence.ordinal,
                "source_file_sha256": item["source_file_sha256"],
                "acquisition_checksum_sha256": item["archive_sha256"],
                "original_text": sentence.original_text,
                "original_text_sha256": text_hash(sentence.original_text)
                if sentence.original_text
                else None,
                "benchmark_text_sha256": text_hash(sentence.text),
                "normalized_text_sha256": text_hash(normalize_text(sentence.text)),
                "source_sentence_annotation_sha256": sentence.annotation_sha256,
                "source_comments": list(sentence.comments),
                "script": script,
                "genre": genre,
                "token_count": sentence.token_count,
                "multiword_token_count": sentence.multiword_count,
                "empty_node_count": sentence.empty_node_count,
                "reconstruction": sentence.reconstruction,
                "warnings": list(sentence.warnings),
                "overlaps": matches,
                "legacy_ud_prior_repository_exposure": index
                in legacy["affected_candidate_indices"],
                "license": "CC-BY-SA-4.0",
                "transformations": [
                    "extract_official_text_or_reconstruct_surface",
                    "preserve_unicode",
                    "classify_script",
                    "audit_overlaps",
                ],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "ud_kurdish.jsonl"
    provenance_path = output_dir / "ud_kurdish.provenance.jsonl"
    _write_jsonl(data_path, (record.to_dict() for record in records))
    _write_jsonl(provenance_path, provenance)
    validation = validate_benchmark(data_path, provenance_path)
    strict_path = output_dir / "ud_kurdish.strict.jsonl"
    strict_provenance_path = output_dir / "ud_kurdish.strict.provenance.jsonl"
    _write_jsonl(
        strict_path, (record.to_dict() for record in records if record.strict_external)
    )
    _write_jsonl(
        strict_provenance_path,
        (
            sidecar
            for record, sidecar in zip(records, provenance, strict=True)
            if record.strict_external
        ),
    )
    if validation["strict_external"]:
        validate_benchmark(strict_path, strict_provenance_path)
    audit = {
        "benchmark": BENCHMARK,
        "release": VERSION,
        "evaluation_only": True,
        "canonical_comparisons_complete": comparisons_complete,
        "legacy_comparison_complete": legacy_comparison_complete,
        "restrictions": [
            "no_training",
            "no_development",
            "no_model_selection",
            "no_calibration",
        ],
        "treebanks": stats,
        "record_count": len(records),
        "strict_external_count": validation["strict_external"],
        "new_langid_eligible_count": sum(
            not (
                {
                    "within_ud_duplicate_text",
                    "duplicate_source_sent_id",
                    "canonical_source_overlap",
                    "canonical_comparison_unavailable",
                    "legacy_ud_comparison_unavailable",
                }
                & set(record.exclusion_reasons)
            )
            for record in records
        ),
        "language_distribution": dict(
            sorted(Counter(record.expected_language for record in records).items())
        ),
        "script_distribution": dict(sorted(script_counts.items())),
        "genre_distribution": dict(sorted(genre_counts.items())),
        "median_character_length": {
            key: statistics.median(values) for key, values in sorted(lengths.items())
        },
        "duplicate_text_records": len(duplicate_indices),
        "duplicate_text_groups": [
            {
                "normalized_text_sha256": text_hash(normalized_text),
                "record_ids": [records[index].id for index in indices],
                "original_splits": [records[index].original_split for index in indices],
                "languages": [records[index].expected_language for index in indices],
            }
            for normalized_text, indices in sorted(normalized_groups.items())
            if len(indices) > 1
        ],
        "cross_original_split_duplicate_groups": sum(
            len({records[index].original_split for index in indices}) > 1
            for indices in normalized_groups.values()
            if len(indices) > 1
        ),
        "duplicate_sent_ids": {
            f"{treebank}:{split}:{sent_id}": count
            for (treebank, split, sent_id), count in sorted(sent_ids.items())
            if count > 1
        },
        "exclusions": dict(sorted(exclusion_counts.items())),
        "cross_source": {**cross["sources"], "near_method": cross["near_method"]},
        "legacy_ud": {
            "files": legacy["files"],
            "affected_records": len(legacy["affected_candidate_indices"]),
        },
        "strict_external_record_ids": [
            record.id for record in records if record.strict_external
        ],
        "excluded_record_ids": {
            record.id: list(record.exclusion_reasons)
            for record in records
            if not record.strict_external
        },
    }
    audit_path = output_dir / "ud_kurdish.audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path = output_dir / "ud_kurdish.audit.txt"
    text_path.write_text(
        f"{BENCHMARK}: {len(records)} records, {validation['strict_external']} strict external.\n"
        + "\n".join(
            f"{language}: {stats[language]['source_sentences']} source, {stats[language]['accepted']} accepted, {len(stats[language]['malformed'])} malformed"
            for language in selected
        )
        + "\nSee audit JSON for overlap IDs, file hashes, scripts and exclusions.\n",
        encoding="utf-8",
    )
    benchmark_manifest = {
        "benchmark": BENCHMARK,
        "evaluation_partition": PARTITION,
        "source_versions": {language: VERSION for language in selected},
        "source_hashes": {
            language: stats[language]["archive_sha256"] for language in selected
        },
        "output_hashes": {
            path.name: sha256_file(path)
            for path in (
                data_path,
                provenance_path,
                strict_path,
                strict_provenance_path,
                audit_path,
                text_path,
            )
        },
        "record_count": len(records),
        "strict_external_count": validation["strict_external"],
        "language_distribution": audit["language_distribution"],
        "script_distribution": audit["script_distribution"],
        "evaluation_restrictions": audit["restrictions"],
    }
    manifest_path = output_dir / "ud_kurdish.manifest.json"
    manifest_path.write_text(
        json.dumps(benchmark_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "audit": audit,
        "manifest": benchmark_manifest,
        "outputs": {
            "data": str(data_path),
            "provenance": str(provenance_path),
            "strict": str(strict_path),
            "strict_provenance": str(strict_provenance_path),
            "audit": str(audit_path),
            "text": str(text_path),
            "manifest": str(manifest_path),
        },
    }

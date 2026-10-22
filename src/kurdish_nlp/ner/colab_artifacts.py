"""Private Colab transfer and interruption-safe artifacts (standard library only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from .build import build, dump, verify
from .dataset import sha256

SOURCE_SHA256 = "31fec7da9e6ec597eaa3977107bf8bece09da0f78fe09da98e06edd1e33f3847"
LOCK_SHA256 = "9cdda285aee64f1119554c40559be45e41799f8a1a19d4c4f53eff37868a8651"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    dump(temporary, value)
    os.replace(temporary, path)


def hashes(root: Path, exclude=()) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are not supported")
        if path.is_file() and path.relative_to(root).as_posix() not in exclude:
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member")
    return str(path)


def prepare_capsule(repo: Path, output: Path) -> dict:
    """Allowlist source only: no raw data, weights, credentials or Git internals."""
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    files = [
        *repo.glob("src/kurdish_nlp/**/*.py"),
        *repo.glob("configs/ner/*"),
        repo / "scripts/ner_download_encoder.py",
        repo / "scripts/ner_colab_environment.py",
        repo / "tests/test_ner_colab_numerics.py",
        repo / "tests/test_ner_colab_dlpack.py",
        repo / "pyproject.toml",
        repo / "README.md",
    ]
    files = sorted(
        p
        for p in files
        if p.is_file() and p.suffix in {".py", ".json", ".cfg", ".txt", ".toml", ".md"}
    )
    manifest = {
        "format": 1,
        "base_git_commit": revision,
        "revision_note": "Base commit plus exact allowlisted working-tree content; includes uncommitted NER preparation.",
        "files": {p.relative_to(repo).as_posix(): sha256(p) for p in files},
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["content_sha256"] = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            info = zipfile.ZipInfo(
                p.relative_to(repo).as_posix(), (2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo("code-manifest.json", (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"archive": str(output), "archive_sha256": sha256(output), **manifest}


def restore_capsule(archive: Path, output: Path, expected_sha: str) -> dict:
    if sha256(archive) != expected_sha:
        raise ValueError("Code capsule checksum mismatch")
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("code-manifest.json"))
        expected = set(manifest["files"]) | {"code-manifest.json"}
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(
            expected
        ):
            raise ValueError("Unexpected / duplicate code members")
        for name in expected:
            safe_name(name)
        if output.exists():
            if any(
                not (output / name).is_file() or sha256(output / name) != digest
                for name, digest in manifest["files"].items()
            ):
                raise ValueError("Existing checkout differs; use a fresh workspace")
            return manifest
        stage = output.with_name(f".{output.name}-{uuid.uuid4().hex}.partial")
        stage.mkdir(parents=True)
        for name in sorted(expected):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read(name))
        if hashes(stage, {"code-manifest.json"}) != manifest["files"]:
            raise ValueError("Code payload differs from its manifest")
        os.replace(stage, output)
    return manifest


def import_source(
    source: Path, target: Path, expected_sha: str = SOURCE_SHA256
) -> None:
    """Accept raw adyan.txt or a zip/tar containing exactly one adyan.txt.

    Read the sole payload directly, never extract untrusted archive paths.
    """
    if target.exists():
        if sha256(target) != expected_sha:
            raise ValueError("Existing original BIO file has wrong checksum")
        return
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            candidates = [
                i
                for i in archive.infolist()
                if not i.is_dir() and PurePosixPath(i.filename).name == "adyan.txt"
            ]
            if len(candidates) != 1 or candidates[0].file_size > 20_000_000:
                raise ValueError(
                    "Source archive must contain exactly one small adyan.txt"
                )
            safe_name(candidates[0].filename)
            payload = archive.read(candidates[0])
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            candidates = [
                i
                for i in archive.getmembers()
                if i.isfile() and PurePosixPath(i.name).name == "adyan.txt"
            ]
            if len(candidates) != 1 or candidates[0].size > 20_000_000:
                raise ValueError(
                    "Source archive must contain exactly one small adyan.txt"
                )
            safe_name(candidates[0].name)
            payload = archive.extractfile(candidates[0]).read()
    else:
        payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(
            "Original BIO checksum mismatch; no alternate dataset is allowed"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def ensure_release(source: Path, data: Path, lock: Path, policy: Path) -> dict:
    if sha256(source) != SOURCE_SHA256 or sha256(lock) != LOCK_SHA256:
        raise ValueError("Official source or release lock changed")
    if not data.exists():
        stage = data.with_name(f".{data.name}-{uuid.uuid4().hex}.partial")
        candidate = stage.with_suffix(".candidate-lock.json")
        build(source, policy, stage, candidate)
        if candidate.read_bytes() != lock.read_bytes():
            raise ValueError("Deterministic rebuild differs from frozen release")
        os.replace(stage, data)
        candidate.unlink()
    return verify(data, lock, source)


def validate_pipeline_files(path: Path, labels: list[str]) -> None:
    required = (
        "config.cfg",
        "meta.json",
        "tokenizer",
        "ner/model",
        "ner/moves",
        "transformer/model",
        "vocab/strings.json",
    )
    if any(
        not (path / name).is_file() or (path / name).stat().st_size == 0
        for name in required
    ):
        raise ValueError("Incomplete serialized transformer pipeline")
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("pipeline") != ["transformer", "ner"] or sorted(
        meta["labels"]["ner"]
    ) != sorted(labels):
        raise ValueError("Checkpoint pipeline/labels differ from the release")


def validate_snapshot(path: Path) -> dict:
    marker = json.loads((path / "COMPLETE.json").read_text())
    if hashes(path, {"COMPLETE.json"}) != marker["files"]:
        raise ValueError(
            "Incomplete or corrupt backup; previous snapshot remains valid"
        )
    for role in ("model-best", "model-last"):
        validate_pipeline_files(path / "checkpoints" / role, marker["labels"])
    return marker


def backup_run(
    run: Path, drive_run: Path, labels: list[str], step: int, *, retain: int = 2
) -> Path:
    """Call synchronously after spaCy finishes saving, never in a polling thread.

    Unique generations + a completion marker also protect against Drive/FUSE
    renames not being durably atomic. Restore always rechecks every byte.
    """
    if retain < 2:
        raise ValueError("Keep at least two complete generations")
    if (drive_run / "COMPLETED.json").exists():
        raise FileExistsError("Completed Drive runs are immutable")
    state = json.loads((run / "run.json").read_text())
    drive_run.mkdir(parents=True, exist_ok=True)
    identity = drive_run / "identity.json"
    wanted = {
        "run_id": state["run_id"],
        "dataset_lock_sha256": state["dataset_lock_sha256"],
    }
    if identity.exists() and json.loads(identity.read_text()) != wanted:
        raise ValueError("Drive run identity collision")
    if not identity.exists():
        atomic_json(identity, wanted)
    for role in ("model-best", "model-last"):
        validate_pipeline_files(run / "checkpoints" / role, labels)
    # At most two full generations plus an in-flight generation: about 6.4 GiB.
    if shutil.disk_usage(drive_run).free < 2.5 * 1024**3:
        raise OSError("Drive needs at least 2.5 GiB free for the next complete backup")
    generation = f"step-{step:08d}-{uuid.uuid4().hex[:8]}"
    stage = drive_run / f".{generation}.partial"
    stage.mkdir()
    for role in ("model-best", "model-last"):
        shutil.copytree(run / "checkpoints" / role, stage / "checkpoints" / role)
    for name in (
        "run.json",
        "config.cfg",
        "effective-changes.json",
        "runtime-policy.json",
        "numerics.jsonl",
        "dependencies.txt",
        "code-manifest.json",
        "environment.json",
        "train.log",
        "metrics.jsonl",
        "selection.lock.json",
        "test-evaluation.json",
        "test-evaluation.receipt.json",
        "preflight-smoke.json",
        "preflight-stress.json",
        "preflight-benchmark.json",
        "evaluation.log",
    ):
        if (run / name).is_file():
            shutil.copy2(run / name, stage / name)
    # Check both copies and original checkpoint bytes while the trainer is paused.
    for role in ("model-best", "model-last"):
        if hashes(run / "checkpoints" / role) != hashes(stage / "checkpoints" / role):
            raise ValueError("Checkpoint changed or copy was incomplete")
    dump(
        stage / "COMPLETE.json",
        {
            "format": 1,
            "run_id": state["run_id"],
            "step": step,
            "labels": labels,
            "files": hashes(stage),
            "resume_state": "weights_only_no_optimizer_or_scheduler",
        },
    )
    validate_snapshot(stage)
    published = drive_run / generation
    os.replace(stage, published)
    # Publish pointers only AFTER a fully checked complete generation exists.
    pointer = {
        "generation": generation,
        "complete_sha256": sha256(published / "COMPLETE.json"),
    }
    atomic_json(drive_run / "latest.json", pointer)
    atomic_json(drive_run / "best.json", {**pointer, "role": "model-best"})
    # Explicit, narrowly scoped retention of generated backup copies, not runs.
    complete = sorted(
        (
            p
            for p in drive_run.glob("step-*")
            if p.is_dir() and (p / "COMPLETE.json").is_file()
        ),
        key=lambda p: (json.loads((p / "COMPLETE.json").read_text())["step"], p.name),
    )
    keep = {published, *[p for p in complete if p != published][-retain + 1 :]}
    for old in complete:
        if old not in keep and re.fullmatch(r"step-\d{8}-[0-9a-f]{8}", old.name):
            if (
                json.loads((old / "COMPLETE.json").read_text())["run_id"]
                != state["run_id"]
            ):
                raise ValueError("Refuse to prune another run")
            shutil.rmtree(old)
    return published


def restore_run(drive_run: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Restore into a new local directory")
    pointer = json.loads((drive_run / "latest.json").read_text())
    generation = safe_name(pointer["generation"])
    if "/" in generation or not re.fullmatch(r"step-\d{8}-[0-9a-f]{8}", generation):
        raise ValueError("Invalid snapshot pointer")
    source = drive_run / generation
    if sha256(source / "COMPLETE.json") != pointer["complete_sha256"]:
        raise ValueError("Backup pointer checksum mismatch")
    validate_snapshot(source)
    stage = output.with_name(f".{output.name}-{uuid.uuid4().hex}.partial")
    shutil.copytree(source, stage)
    validate_snapshot(stage)
    os.replace(stage, output)
    # A completed test report/receipt may have reached Drive before its large
    # checkpoint snapshot when a session disconnected. Recover those small,
    # hash-bound artifacts rather than accidentally testing again.
    report_path, receipt_path = (
        drive_run / "test-evaluation.json",
        drive_run / "test-evaluation.receipt.json",
    )
    if report_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            not (output / "selection.lock.json").exists()
            or receipt["report_sha256"] != sha256(report_path)
            or receipt["selection_sha256"] != sha256(output / "selection.lock.json")
        ):
            raise ValueError("Recovered evaluation is not bound to this selection")
        shutil.copy2(report_path, output / "test-evaluation.json")
        shutil.copy2(receipt_path, output / "test-evaluation.receipt.json")
        state = json.loads((output / "run.json").read_text())
        state["test_predictions_made"] = True
        atomic_json(output / "run.json", state)
    return output

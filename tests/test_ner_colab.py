"""Offline Colab transfer, transactional backup and notebook safety regressions."""

import ast
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from kurdish_nlp.ner.colab_artifacts import (
    LOCK_SHA256,
    SOURCE_SHA256,
    atomic_json,
    backup_run,
    ensure_release,
    hashes,
    import_source,
    prepare_capsule,
    restore_capsule,
    restore_run,
    validate_snapshot,
)

REPO = Path(__file__).resolve().parents[1]


def test_real_frozen_identities_are_unchanged():
    assert (
        hashlib.sha256((REPO / "configs/ner/ner-v2.lock.json").read_bytes()).hexdigest()
        == LOCK_SHA256
    )
    lock = json.loads((REPO / "configs/ner/ner-v2.lock.json").read_text())
    assert lock["source_sha256"] == SOURCE_SHA256
    assert lock["segments"] == {"train": 17396, "dev": 2058, "test": 2145}
    assert len(lock["labels"]) == 15


@pytest.mark.parametrize("kind", ["raw", "zip", "tar"])
def test_source_import_preserves_exact_bytes_and_rejects_bad_digest(tmp_path, kind):
    payload = "ناسا B-ORG\n".encode()
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / f"upload.{kind}"
    if kind == "raw":
        source.write_bytes(payload)
    elif kind == "zip":
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("private/adyan.txt", payload)
    else:
        with tarfile.open(source, "w") as archive:
            info = tarfile.TarInfo("private/adyan.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    target = tmp_path / "raw/adyan.txt"
    with pytest.raises(ValueError, match="checksum"):
        import_source(source, target)
    assert not target.exists()
    import_source(source, target, digest)
    assert target.read_bytes() == payload
    import_source(source, target, digest)  # Idempotent restoration only if unchanged.
    target.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        import_source(source, target, digest)


def test_source_archive_ambiguity_and_traversal_rejected(tmp_path):
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../adyan.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        import_source(source, tmp_path / "out")
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("a/adyan.txt", "bad")
        archive.writestr("b/adyan.txt", "bad")
    with pytest.raises(ValueError, match="exactly one"):
        import_source(source, tmp_path / "out")


def test_bad_source_refuses_release_before_rebuild(tmp_path):
    source = tmp_path / "adyan.txt"
    source.write_text("wrong")
    with pytest.raises(ValueError, match="changed"):
        ensure_release(
            source,
            tmp_path / "release",
            REPO / "configs/ner/ner-v2.lock.json",
            REPO / "configs/ner/policy-v2.json",
        )
    assert not (tmp_path / "release").exists()


def _run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    atomic_json(
        run / "run.json",
        {"run_id": "sorani-ner-v2-fixture", "dataset_lock_sha256": "fixture"},
    )
    for role in ("model-best", "model-last"):
        model = run / "checkpoints" / role
        for name in (
            "config.cfg",
            "tokenizer",
            "ner/model",
            "ner/moves",
            "transformer/model",
            "vocab/strings.json",
        ):
            p = model / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("serialization fixture")
        atomic_json(
            model / "meta.json",
            {"pipeline": ["transformer", "ner"], "labels": {"ner": ["PER"]}},
        )
    return run


def test_atomic_backup_retention_restore_and_completed_guard(tmp_path):
    run = _run(tmp_path)
    drive = tmp_path / "drive"
    for step in range(4):
        (run / "checkpoints/model-last/ner/model").write_text(f"weights fixture {step}")
        backup_run(run, drive, ["PER"], step)
    assert len(list(drive.glob("step-*"))) == 2
    restored = restore_run(drive, tmp_path / "restored")
    validate_snapshot(restored)
    assert (
        restored / "checkpoints/model-last/ner/model"
    ).read_text() == "weights fixture 3"
    assert json.loads((drive / "best.json").read_text())["role"] == "model-best"
    with pytest.raises(FileExistsError):
        restore_run(drive, restored)
    atomic_json(drive / "COMPLETED.json", {"done": True})
    with pytest.raises(FileExistsError, match="immutable"):
        backup_run(run, drive, ["PER"], 5)


def test_partial_copy_never_replaces_latest_good_backup(tmp_path, monkeypatch):
    import kurdish_nlp.ner.colab_artifacts as module

    run, drive = _run(tmp_path), tmp_path / "drive"
    good = backup_run(run, drive, ["PER"], 1)
    pointer = (drive / "latest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("Simulated Colab/Drive disconnect")

    monkeypatch.setattr(module.shutil, "copytree", fail)
    with pytest.raises(OSError, match="disconnect"):
        backup_run(run, drive, ["PER"], 2)
    assert (drive / "latest.json").read_bytes() == pointer
    validate_snapshot(good)
    (good / "checkpoints/model-last/ner/model").write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        validate_snapshot(good)


def test_code_capsule_allowlist_manifest_and_existing_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # No commit needed: use a repository HEAD from the real checkout for this test.
    real_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    (repo / ".git/HEAD").write_text(real_head + "\n")
    source = repo / "src/kurdish_nlp/ner/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    private = repo / "data/ner/raw/adyan.txt"
    private.parent.mkdir(parents=True)
    private.write_text("must not transfer")
    archive = tmp_path / "code.zip"
    result = prepare_capsule(repo, archive)
    replay = prepare_capsule(repo, tmp_path / "replayed-code.zip")
    assert replay["archive_sha256"] == result["archive_sha256"]
    assert all(not n.startswith(("data/", ".git/")) for n in result["files"])
    out = tmp_path / "checkout"
    restore_capsule(archive, out, result["archive_sha256"])
    assert hashes(out, {"code-manifest.json"}) == result["files"]
    (out / "generated-cache").write_text("permitted")
    restore_capsule(archive, out, result["archive_sha256"])
    (out / "src/kurdish_nlp/ner/fixture.py").write_text("tamper")
    with pytest.raises(ValueError, match="differs"):
        restore_capsule(archive, out, result["archive_sha256"])


def test_cuda_commands_refuse_local_non_linux_host(monkeypatch):
    from kurdish_nlp.ner import colab

    monkeypatch.setattr(colab.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="do not run it on the Mac"):
        colab.gpu_environment()


def test_notebook_order_json_and_all_python_cells_compile():
    nb = json.loads((REPO / "notebooks/sorani_ner_v2_colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    sections = []
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and text.startswith("## "):
            sections.append(int(text.split()[1].rstrip(".")))
        if cell["cell_type"] == "code":
            ast.parse(text)
            assert cell["execution_count"] is None
            assert not cell["outputs"]
    assert sections == list(range(1, 16))
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assignments = {
        target.id: node.value.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["RUN_FULL_TRAINING"] is False
    assert assignments["EVALUATE_FROZEN_TEST"] is False
    assert "EXPECTED_CODE_SHA256" in code


def test_training_gate_requires_all_matching_passed_reports(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kurdish_nlp.ner import colab

    state = {"config_sha256": "frozen", "runtime_policy_sha256": "policy"}
    cfg = {"components": {"transformer": {"model": {"mixed_precision": False}}}}
    monkeypatch.setattr(colab, "load_run", lambda args: (state, cfg, {}))
    args = SimpleNamespace(run=tmp_path)
    with pytest.raises(FileNotFoundError):
        colab.gate(args)
    for stage in ("smoke", "stress", "benchmark"):
        atomic_json(
            tmp_path / f"preflight-{stage}.json",
            {
                "status": "passed",
                "config_sha256": "frozen",
                "runtime_policy_sha256": "policy",
                "updates": [{"optimizer_updates": 1}],
            },
        )
    assert colab.gate(args) == state
    atomic_json(
        tmp_path / "preflight-stress.json",
        {"status": "failed", "config_sha256": "frozen"},
    )
    with pytest.raises(ValueError, match="preflights"):
        colab.gate(args)

    atomic_json(
        tmp_path / "preflight-stress.json",
        {"status": "passed", "config_sha256": "changed"},
    )
    with pytest.raises(ValueError, match="preflights"):
        colab.gate(args)

    for bad in (
        {"runtime_policy_sha256": "different"},
        {"updates": [{"optimizer_updates": 0}]},
        {"updates": []},
    ):
        atomic_json(
            tmp_path / "preflight-stress.json",
            {
                "status": "passed",
                "config_sha256": "frozen",
                "runtime_policy_sha256": "policy",
                "updates": [{"optimizer_updates": 1}],
                **bad,
            },
        )
        with pytest.raises(ValueError, match="preflights"):
            colab.gate(args)


def test_spacy_registry_and_portable_saved_logger_config(tmp_path):
    spacy = pytest.importorskip("spacy")
    from spacy.util import load_config, registry

    from kurdish_nlp.ner.colab import register_logger

    register_logger()
    register_logger()
    cfg = load_config(REPO / "configs/ner/xlmr-v2.cfg")
    changed = cfg.copy()
    changed["training"]["gpu_allocator"] = "pytorch"
    assert cfg["training"]["gpu_allocator"] is None
    log_cfg = {
        "@loggers": "kurdish_nlp.ColabCheckpointLogger.v1",
        "run_path": str(tmp_path / "run"),
        "drive_path": str(tmp_path / "drive"),
    }
    assert callable(registry.resolve({"logger": log_cfg})["logger"])
    nlp = spacy.blank("xx")
    nlp.config["training"]["logger"] = log_cfg
    nlp.to_disk(tmp_path / "portable")
    # A saved training logger does not need to run during inference.
    assert spacy.load(tmp_path / "portable")("😀 سڵاو").text == "😀 سڵاو"


def test_spacy_small_batch_accumulation_is_bounded_without_losing_records(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("spacy")
    from spacy.training import loop

    from kurdish_nlp.ner.colab import install_microbatch_guard

    original = loop.subdivide_batch
    monkeypatch.setattr(loop, "subdivide_batch", original)
    batch = [SimpleNamespace(predicted=[i]) for i in range(7)]
    assert len(next(iter(original(batch, 16)))) == 7  # Pinned upstream edge case.
    install_microbatch_guard()
    install_microbatch_guard()
    result = list(loop.subdivide_batch(batch, 16))
    assert all(len(group) == 1 for group in result)
    assert [record for group in result for record in group] == batch
    for size in (4, 8, 12):
        records = [SimpleNamespace(predicted=[i]) for i in range(size)]
        assert list(loop.subdivide_batch(records, 4)) == list(original(records, 4))


def test_cuda_libraries_are_discovered_before_framework_imports(tmp_path):
    from kurdish_nlp.ner.cuda_runtime import runtime_environment

    root = tmp_path / "site-packages"
    for name in (
        "nvidia/cublas/lib",
        "nvidia/cuda_nvrtc/lib",
        "torch/lib",
        "nvidia/cuda_runtime/lib",
    ):
        (root / name).mkdir(parents=True)
    include = root / "nvidia/cuda_runtime/include"
    include.mkdir()
    (include / "cuda_runtime.h").touch()
    env = runtime_environment({"LD_LIBRARY_PATH": "/existing", "KEEP": "yes"}, [root])
    assert env["LD_LIBRARY_PATH"].split(":")[-1] == "/existing"
    assert str(root / "nvidia/cuda_nvrtc/lib") in env["LD_LIBRARY_PATH"].split(":")
    assert env["CUDA_PATH"] == str(root / "nvidia/cuda_runtime")
    assert runtime_environment(env, [root]) == env  # No re-exec loop.
    assert env["KEEP"] == "yes"


def test_encoder_download_creates_pointer_directory_with_external_hf_home(
    tmp_path, monkeypatch
):
    import runpy
    import sys
    from types import SimpleNamespace

    from kurdish_nlp.ner import training

    config = tmp_path / "configs/ner"
    config.mkdir(parents=True)
    (config / "encoder-v2.json").write_text(
        json.dumps({"repo_id": "fixture", "revision": "pinned", "files": {}})
    )
    downloaded = tmp_path / "drive-cache/snapshot"
    downloaded.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "drive-cache"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda *a, **kw: str(downloaded)),
    )
    monkeypatch.setattr(training, "validate_encoder", lambda path: None)
    runpy.run_path(str(REPO / "scripts/ner_download_encoder.py"), run_name="__main__")
    assert (tmp_path / "data/ner/cache/encoder-path.txt").read_text().strip() == str(
        downloaded
    )


@pytest.mark.parametrize("size,accumulate", [(15, 8), (7, 16), (31, 8), (17, 4)])
def test_uneven_microbatch_remainders_are_bounded(size, accumulate):
    from types import SimpleNamespace

    pytest.importorskip("spacy")
    from spacy.training import loop

    from kurdish_nlp.ner.colab import install_microbatch_guard

    install_microbatch_guard()
    records = [SimpleNamespace(predicted=[i]) for i in range(size)]
    groups = list(loop.subdivide_batch(records, accumulate))
    assert max(map(len, groups)) <= max(1, size // accumulate)
    assert [r for group in groups for r in group] == records

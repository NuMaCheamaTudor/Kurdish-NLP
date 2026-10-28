"""No-ML checkpoint transaction, immutable-source and compatibility tests."""

import copy
import hashlib
import json
import shutil

import pytest
from test_ner_colab import _run

from kurdish_nlp.ner.colab_artifacts import (
    atomic_json,
    backup_run,
    hashes,
    restore_run,
    validate_code_manifest,
)
from kurdish_nlp.ner.continued import verified_source
from kurdish_nlp.ner.training_state import (
    latest_generation,
    publish_generation,
    restore_training,
    validate_generation,
    versions,
)


def fixture_writer(run):
    def writer(stage):
        shutil.copytree(run / "checkpoints", stage / "checkpoints", dirs_exist_ok=True)
        atomic_json(
            stage / "training-state/state.json",
            {
                "format": 2,
                "versions": versions(),
                "progress": {"updates": 1},
                "scaler": {"_found_inf": False},
            },
        )

    return writer


def test_restore_exact_step6000_generation_leaves_original_backup_immutable(tmp_path):
    run = _run(tmp_path)
    drive = tmp_path / "drive"
    source = backup_run(run, drive, ["PER"], 6000)
    backup_run(run, drive, ["PER"], 6200)
    before = hashes(drive)
    output = tmp_path / source.name
    restore_run(drive, output, generation=source.name)
    assert hashes(output) == hashes(source)
    model, marker = verified_source(
        output, "model-last", run_id="sorani-ner-v2-fixture", generation=source.name
    )
    assert marker["step"] == 6000 and model.name == "model-last"
    with pytest.raises(ValueError, match="Source run"):
        verified_source(output, "model-last", run_id="different")
    with pytest.raises(ValueError, match="generation"):
        verified_source(output, "model-last", generation="wrong")
    with pytest.raises(ValueError, match="Unsafe"):
        restore_run(drive, tmp_path / "bad", generation="../invalid")
    assert hashes(drive) == before


def test_atomic_generation_pointer_failure_preserves_verified_previous(
    tmp_path, monkeypatch
):
    import kurdish_nlp.ner.training_state as module

    run, root = _run(tmp_path), tmp_path / "state"
    meta = {"identity": {"run_id": "fixture"}, "updates": 1, "labels": ["PER"]}
    good = publish_generation(root, fixture_writer(run), meta)
    before = hashes(good)
    pointer = (root / "latest.json").read_bytes()
    original = module.atomic_json

    def disconnect(path, value):
        if path.name == "latest.json":
            raise OSError("Drive disconnected before pointer publication")
        return original(path, value)

    monkeypatch.setattr(module, "atomic_json", disconnect)
    with pytest.raises(OSError, match="disconnected"):
        publish_generation(root, fixture_writer(run), meta)
    assert (root / "latest.json").read_bytes() == pointer
    assert hashes(good) == before
    assert latest_generation(root) == good
    assert len(list(root.glob("update-*"))) == 2


def test_generation_rejects_changed_bytes_stack_and_weights_only(tmp_path):
    run, root = _run(tmp_path), tmp_path / "state"
    meta = {"identity": {"run_id": "fixture"}, "updates": 1, "labels": ["PER"]}
    good = publish_generation(root, fixture_writer(run), meta)
    (good / "training-state/state.json").write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        restore_training(root, tmp_path / "blocked")
    with pytest.raises(ValueError, match="weights-only"):
        old = backup_run(run, tmp_path / "legacy", ["PER"], 6000)
        validate_generation(old)

    def wrong_stack(stage):
        fixture_writer(run)(stage)
        path = stage / "training-state/state.json"
        data = json.loads(path.read_text())
        data["versions"]["spacy"] = "0.0"
        atomic_json(path, data)

    with pytest.raises(ValueError, match="stack"):
        publish_generation(root, wrong_stack, meta)


def test_source_capsule_identity_cannot_be_relaxed_by_removing_entries(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "module.py").write_text("unchanged\n")
    manifest = {
        "format": 1,
        "files": {"module.py": hashlib.sha256(b"unchanged\n").hexdigest()},
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    validate_code_manifest(manifest)
    changed = copy.deepcopy(manifest)
    changed["files"] = {}
    with pytest.raises(ValueError, match="identity"):
        validate_code_manifest(changed)
    (tmp_path / "module.py").write_text("changed\n")
    with pytest.raises(ValueError, match="checkout"):
        validate_code_manifest(manifest)


def test_environment_installer_accepts_exact_resume_python_patch(tmp_path, monkeypatch):
    import platform
    import runpy
    import subprocess
    from pathlib import Path

    install = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/ner_colab_environment.py")
    )["install"]
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kw: commands.append(command)
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kw: "[3, 12, 14]" if "-c" in command else "Python 3.12.14\n",
    )
    assert install(tmp_path, environment, "3.12.14") == environment / "bin/python"
    assert (
        len(commands) == 2
    )  # Pinned install and pip check, no interpreter replacement.
    with pytest.raises(RuntimeError, match="incompatible"):
        install(tmp_path, environment, "3.12.13")
    with pytest.raises(ValueError, match="Python 3.12"):
        install(tmp_path, environment, "3.13")


def test_checkpoint_hashing_uses_bounded_reads_without_changing_digest(tmp_path):
    from kurdish_nlp.ner.dataset import sha256

    payload = b"checkpoint bytes" * 1_000_000
    path = tmp_path / "large-model"
    path.write_bytes(payload)

    class BoundedPath:
        def open(self, mode):
            raw = path.open(mode)

            class Reader:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    raw.close()

                def read(self, size):
                    assert 0 < size <= 8 * 1024**2
                    return raw.read(size)

            return Reader()

        def read_bytes(self):
            pytest.fail("Unbounded checkpoint read")

    assert sha256(BoundedPath()) == hashlib.sha256(payload).hexdigest()

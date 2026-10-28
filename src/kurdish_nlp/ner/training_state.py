"""Versioned, pickle-free state for the audited spaCy/Thinc training stack.

Only completed effective batches are recoverable. Uncommitted work is replayed
from the last published generation; no partial accumulation is ever restored.
"""

from __future__ import annotations

import base64
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import uuid

from .colab_artifacts import atomic_json, hashes, safe_name, validate_pipeline_files
from .dataset import sha256

FORMAT = 2
PACKAGES = (
    "spacy",
    "thinc",
    "spacy-transformers",
    "transformers",
    "torch",
    "numpy",
    "cupy-cuda12x",
    "tokenizers",
    "sentencepiece",
    "safetensors",
    "srsly",
    "blis",
    "cymem",
    "preshed",
    "murmurhash",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
)


def versions():
    result = {"python": list(sys.version_info[:3])}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def execution_signature(cuda):
    import torch

    return {
        "cuda": torch.version.cuda if cuda else None,
        "devices": [
            [torch.cuda.get_device_name(i), list(torch.cuda.get_device_capability(i))]
            for i in range(torch.cuda.device_count())
        ]
        if cuda
        else [],
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def learning_rate(schedule, completed):
    """Positive first warmup update; final update also has a positive LR.

    `completed` is the number of committed optimizer updates, not spaCy's
    historical zero-based log step. No generator needs to be pickled/replayed.
    """
    peak, warmup, total = (schedule[k] for k in ("peak", "warmup", "total"))
    if not (math.isfinite(peak) and 0 < peak <= 2e-5 and 0 <= warmup < total):
        raise ValueError("Invalid continued-training learning-rate schedule")
    if not 0 <= completed < total:
        raise ValueError("Schedule exhausted")
    factor = (
        (completed + 1) / warmup
        if completed < warmup
        else (total - completed) / (total - warmup)
    )
    return peak * factor


def fresh_optimizer(nlp, schedule):
    from spacy.util import registry

    cfg = dict(nlp.config["training"]["optimizer"])
    if cfg.get("@optimizers") != "Adam.v1":
        raise ValueError("Stateful training requires the audited Adam.v1 optimizer")
    cfg["learn_rate"] = learning_rate(schedule, 0)
    optimizer = registry.resolve({"optimizer": cfg})["optimizer"]
    if optimizer.averages is not None or optimizer.use_radam or optimizer.schedules:
        raise ValueError(
            "Stateful training requires Adam, no averages, scalar optimizer attributes"
        )
    return optimizer


def parameter_map(nlp):
    """Map ephemeral Thinc node/shim IDs to stable graph positions and names."""
    result, topology = {}, []
    for component_name, component in nlp.pipeline:
        for index, node in enumerate(component.model.walk()):
            prefix = f"{component_name}/{index}/{node.name}"
            for name in node.param_names:
                if node.has_param(name):
                    value = node.get_param(name)
                    key = (node.id, name)
                    result[key] = f"{prefix}/param/{name}"
                    topology.append([result[key], list(value.shape), str(value.dtype)])
            for shim_index, shim in enumerate(node.shims):
                if not hasattr(shim, "_model"):
                    raise ValueError("Unsupported non-PyTorch shim")
                for name, value in shim._model.named_parameters():
                    key = (shim.id, name)
                    result[key] = f"{prefix}/shim/{shim_index}/{name}"
                    topology.append([result[key], list(value.shape), str(value.dtype)])
    if len(set(result.values())) != len(result):
        raise ValueError("Ambiguous optimizer parameter mapping")
    return result, topology


def weight_fingerprints(nlp):
    """Hash actual parameter values, independently of serialization metadata."""
    import hashlib

    from thinc.util import to_numpy

    output = {}
    for component_name, component in nlp.pipeline:
        digest = hashlib.sha256()
        for node in component.model.walk():
            for name in node.param_names:
                if node.has_param(name):
                    digest.update(to_numpy(node.get_param(name)).tobytes())
            for shim in node.shims:
                for name, value in shim._model.named_parameters():
                    digest.update(name.encode())
                    for part in value.detach().reshape(-1).split(1048576):
                        digest.update(memoryview(part.cpu().numpy()).cast("B"))
        output[component_name] = digest.hexdigest()
    return output


def _b64(value):
    return base64.b64encode(value).decode("ascii")


def rng_state(cuda):
    import numpy as np
    import torch

    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], _b64(np_state[1].tobytes()), *np_state[2:]],
        "torch": _b64(torch.get_rng_state().numpy().tobytes()),
        "cuda": [
            _b64(t.cpu().numpy().tobytes()) for t in torch.cuda.get_rng_state_all()
        ]
        if cuda
        else [],
        # checked_update seeds CuPy from the saved NumPy/Torch states before
        # EVERY attempt. CuPy's opaque cuRAND handle is never consumed outside
        # that step in this loop; its next seed is the complete required state.
        "cupy": "reseed_from_numpy_and_torch_before_each_effective_batch",
    }


def restore_rng(state):
    import numpy as np
    import torch

    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value

    random.setstate(tuples(state["python"]))
    n = state["numpy"]
    np.random.set_state(
        (n[0], np.frombuffer(base64.b64decode(n[1]), dtype=np.uint32).copy(), *n[2:])
    )
    tensor = lambda x: torch.from_numpy(
        np.frombuffer(base64.b64decode(x), dtype=np.uint8).copy()
    )
    torch.set_rng_state(tensor(state["torch"]))
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all([tensor(x) for x in state["cuda"]])


def _scaler(nlp):
    return nlp.get_pipe("transformer").model.layers[0].shims[0]._grad_scaler


def save_state(path, nlp, optimizer, progress, *, cuda):
    import numpy as np
    import torch
    from thinc.util import to_numpy

    if (
        getattr(nlp, "_ner_v2_in_update", False)
        or getattr(nlp, "_ner_v2_update_index", 0) != progress["updates"]
    ):
        raise ValueError("Cannot checkpoint an incomplete effective-batch update")
    for _, component in nlp.pipeline:
        for node in component.model.walk():
            if any(
                node.has_grad(name)
                and bool(node.ops.xp.count_nonzero(node.get_grad(name)))
                for name in node.param_names
            ):
                raise ValueError("Cannot checkpoint accumulated gradients")
            for shim in node.shims:
                if any(
                    p.grad is not None and bool(torch.count_nonzero(p.grad))
                    for p in shim._model.parameters()
                ):
                    raise ValueError("Cannot checkpoint accumulated gradients")
    path.mkdir()
    keys, topology = parameter_map(nlp)
    slots = {}
    for slot in ("mom1", "mom2", "nr_update", "last_seen"):
        entries = []
        for key, value in getattr(optimizer, slot).items():
            if key not in keys:
                raise ValueError("Optimizer contains an unknown parameter")
            if slot in {"mom1", "mom2"}:
                filename = f"{slot}-{len(entries):05d}.npy"
                np.save(path / filename, to_numpy(value), allow_pickle=False)
                entries.append([keys[key], filename])
            else:
                entries.append([keys[key], int(value)])
        slots[slot] = entries
    scaler = _scaler(nlp)
    scalar_names = (
        "grad_clip",
        "learn_rate",
        "b1",
        "b2",
        "eps",
        "L2",
        "use_radam",
        "L2_is_weight_decay",
    )
    scaler_names = (
        "_enabled",
        "_growth_factor",
        "_backoff_factor",
        "_growth_interval",
        "_found_inf",
    )
    if optimizer.schedules or optimizer.averages is not None or optimizer.use_radam:
        raise ValueError("Unsupported optimizer state")
    if scaler.found_inf:
        raise ValueError("Cannot checkpoint a partial/overflowed effective batch")
    atomic_json(
        path / "state.json",
        {
            "format": FORMAT,
            "versions": versions(),
            "execution": execution_signature(cuda),
            "topology": topology,
            "optimizer": {name: getattr(optimizer, name) for name in scalar_names},
            "slots": slots,
            "scaler": {
                **{name: getattr(scaler, name) for name in scaler_names},
                "_scale": scaler._scale.item(),
                "_growth_tracker": scaler._growth_tracker.item(),
            },
            "progress": progress,
            "rng": rng_state(cuda),
            "update_index": getattr(nlp, "_ner_v2_update_index", 0),
            "accumulation": "empty_at_completed_effective_batch_boundary",
        },
    )


def load_state(path, nlp, optimizer):
    import numpy as np
    from thinc.api import get_current_ops

    state = json.loads((path / "state.json").read_text())
    keys, topology = parameter_map(nlp)
    if (
        state["format"] != FORMAT
        or state["versions"] != versions()
        or state["topology"] != topology
        or state["execution"] != execution_signature(bool(state["rng"]["cuda"]))
    ):
        raise ValueError("Training stack or parameter topology differs from checkpoint")
    if state["accumulation"] != "empty_at_completed_effective_batch_boundary":
        raise ValueError("Partial accumulation cannot be resumed")
    if state["update_index"] != state["progress"]["updates"]:
        raise ValueError("Numerical update index differs from global update count")
    reverse = {v: k for k, v in keys.items()}
    shapes = {entry[0]: entry[1] for entry in topology}
    if set(state["slots"]) != {"mom1", "mom2", "nr_update", "last_seen"}:
        raise ValueError("Incomplete optimizer slots")
    slot_keys = {
        slot: [entry[0] for entry in entries]
        for slot, entries in state["slots"].items()
    }
    if any(len(names) != len(set(names)) for names in slot_keys.values()) or not (
        set(slot_keys["mom1"]) == set(slot_keys["mom2"]) == set(slot_keys["nr_update"])
    ):
        raise ValueError("Incomplete or duplicate optimizer moments")
    for slot, entries in state["slots"].items():
        target = getattr(optimizer, slot)
        target.clear()
        for name, value in entries:
            if name not in reverse:
                raise ValueError("Unknown optimizer parameter")
            if slot in {"mom1", "mom2"}:
                safe_name(value)
                value = get_current_ops().asarray(
                    np.load(path / value, allow_pickle=False)
                )
                if (
                    value.size != math.prod(shapes[name])
                    or str(value.dtype) != "float32"
                    or value.ndim != 1
                ):
                    raise ValueError("Optimizer moment shape/dtype mismatch")
                if not bool(get_current_ops().xp.isfinite(value).all()):
                    raise ValueError("Nonfinite optimizer moments")
            elif not isinstance(value, int) or value < 0:
                raise ValueError("Invalid optimizer update count")
            target[reverse[name]] = value
    for name, value in state["optimizer"].items():
        setattr(optimizer, name, value)
    scaler = _scaler(nlp)
    if (
        not math.isfinite(state["scaler"]["_scale"])
        or state["scaler"]["_scale"] <= 0
        or state["scaler"]["_found_inf"]
        or state["scaler"]["_growth_tracker"] < 0
    ):
        raise ValueError("Invalid FP16 scaler state")
    if scaler._enabled != state["scaler"]["_enabled"]:
        raise ValueError("FP16 mode differs from checkpoint")
    for name, value in state["scaler"].items():
        if name in {"_scale", "_growth_tracker"}:
            getattr(scaler, name).fill_(value)
        else:
            setattr(scaler, name, value)
    nlp._ner_v2_update_index = state["update_index"]
    restore_rng(state["rng"])  # LAST: model load and optimizer allocation consume RNG.
    return state["progress"]


def validate_generation(path, identity=None):
    marker = json.loads((path / "COMPLETE.json").read_text())
    if (
        marker.get("format") != FORMAT
        or marker.get("resume_state") != "full_training_state"
    ):
        raise ValueError(
            "Not a true-resume checkpoint; weights-only requires prepare-continued"
        )
    if hashes(path, {"COMPLETE.json"}) != marker["files"]:
        raise ValueError("Incomplete or corrupt training generation")
    if identity is not None and marker["identity"] != identity:
        raise ValueError("Run/dataset/configuration compatibility mismatch")
    for role in ("model-last", "model-best"):
        validate_pipeline_files(path / "checkpoints" / role, marker["labels"])
    state = json.loads((path / "training-state/state.json").read_text())
    if state["format"] != FORMAT or state["versions"] != versions():
        raise ValueError("Incompatible training stack")
    if (
        state["progress"]["updates"] != marker["updates"]
        or state["scaler"]["_found_inf"]
    ):
        raise ValueError("Invalid training-state boundary")
    return marker


def latest_generation(root):
    pointer = json.loads((root / "latest.json").read_text())
    name = safe_name(pointer["generation"])
    if "/" in name or not name.startswith("update-"):
        raise ValueError("Invalid generation pointer")
    path = root / name
    if sha256(path / "COMPLETE.json") != pointer["complete_sha256"]:
        raise ValueError("Generation pointer checksum mismatch")
    validate_generation(path)
    return path


def publish_generation(root, writer, metadata):
    """Unique immutable generations; preserve ALL verified previous generations.

    COMPLETE is written last; pointers are updated only after byte verification.
    Drive/FUSE consumers must always validate, regardless of rename semantics.
    """
    root.mkdir(parents=True, exist_ok=True)
    name = f"update-{metadata['updates']:08d}-{uuid.uuid4().hex[:12]}"
    stage = root / f".{name}.partial"
    stage.mkdir()
    (stage / "checkpoints").mkdir()
    writer(stage)
    # Flush closed model/array files before publishing the completion marker.
    # FUSE durability still depends on the provider; restore rechecks every byte.
    for path in stage.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    atomic_json(
        stage / "COMPLETE.json",
        {
            **metadata,
            "format": FORMAT,
            "resume_state": "full_training_state",
            "files": hashes(stage, {"COMPLETE.json"}),
        },
    )
    validate_generation(stage, metadata["identity"])
    published = root / name
    os.replace(stage, published)
    atomic_json(
        root / "latest.json",
        {"generation": name, "complete_sha256": sha256(published / "COMPLETE.json")},
    )
    return published


def replicate_generation(source, root):
    marker = validate_generation(source)
    return publish_generation(
        root,
        lambda stage: shutil.copytree(source, stage, dirs_exist_ok=True),
        {
            k: v
            for k, v in marker.items()
            if k not in {"files", "format", "resume_state"}
        },
    )


def restore_training(root, output):
    if output.exists():
        raise FileExistsError("Restore into a fresh local directory")
    source = root if (root / "COMPLETE.json").is_file() else latest_generation(root)
    validate_generation(source)
    stage = output.with_name(f".{output.name}-{uuid.uuid4().hex}.partial")
    sealed = stage / "state-checkpoints" / source.name
    shutil.copytree(source, sealed)
    validate_generation(sealed)
    # Working metadata/corpora/convenience models may change; the recovery
    # generation stays sealed. Never rehydrate data inside a checksummed tree.
    for path in source.iterdir():
        if path.is_file() and path.name != "COMPLETE.json":
            shutil.copy2(path, stage / path.name)
    shutil.copytree(source / "checkpoints", stage / "checkpoints")
    atomic_json(
        stage / "state-checkpoints/latest.json",
        {
            "generation": source.name,
            "complete_sha256": sha256(sealed / "COMPLETE.json"),
        },
    )
    os.replace(stage, output)
    return output

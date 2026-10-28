"""T4-only preflight and guarded spaCy training; every command runs in a venv.

This adds orchestration around the frozen release and existing conversion,
selection and evaluation. It never starts training by being imported.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .build import dump, read_jsonl, verify
from .colab_artifacts import (
    LOCK_SHA256,
    atomic_json,
    backup_run,
    hashes,
    validate_code_manifest,
    validate_pipeline_files,
)
from .colab_numerics import (
    NumericalFailure,
    checked_update,
    configure_model,
    install_training_loop,
    install_window_guard,
)
from .contract import entities_payload
from .dataset import sha256, spans
from .training import convert_records, gold_doc, validate_encoder, wordpiece_check


def gpu_environment() -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "This command requires the Colab Linux x86_64 CUDA runtime; do not run it on the Mac"
        )
    import cupy
    import psutil
    import spacy
    import torch
    from thinc.api import get_current_ops, set_gpu_allocator

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Use the isolated Python 3.12 training interpreter (NetworkX 3.7 requires 3.12)"
        )
    if not torch.cuda.is_available() or torch.version.cuda != "12.4":
        raise RuntimeError("Expected PyTorch 2.6.0+cu124 with usable CUDA 12.4")
    if torch.__version__ != "2.6.0+cu124" or spacy.__version__ != "3.8.7":
        raise RuntimeError("The pinned model stack has changed")
    properties = torch.cuda.get_device_properties(0)
    if "T4" not in properties.name or properties.total_memory < 14 * 1024**3:
        raise RuntimeError("This preflight is scoped to a T4 with at least 14 GiB VRAM")
    spacy.require_gpu(0)
    set_gpu_allocator("pytorch")
    if get_current_ops().xp is not cupy:
        raise RuntimeError("spaCy/Thinc is not using CuPy on CUDA")
    # Compile/run a real CuPy operation, rather than just checking availability.
    if int(cupy.sum(cupy.arange(4)).get()) != 6:
        raise RuntimeError("CuPy CUDA kernel validation failed")
    # main() has completed bootstrap (including any execve). Install in this
    # final process for every CUDA command, before pipeline init/load/backward.
    from .colab_dlpack import install_cupy_dlpack_compatibility

    dlpack_compatibility = install_cupy_dlpack_compatibility()
    return {
        "gpu": properties.name,
        "vram_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cupy": cupy.__version__,
        "spacy": spacy.__version__,
        "thinc_backend": get_current_ops().xp.__name__,
        "dlpack_compatibility": dlpack_compatibility,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_ram_available_bytes": psutil.virtual_memory().available,
        "cpu_rss_bytes": psutil.Process().memory_info().rss,
        "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }


def cuda_model(nlp):
    import torch

    model = nlp.get_pipe("transformer").model.transformer
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("XLM-R weights are not all on CUDA")
    if nlp.get_pipe("ner").model.ops.xp.__name__ != "cupy":
        raise RuntimeError("spaCy NER head is not using CUDA/CuPy")
    torch.cuda.synchronize()
    return model


def memory() -> dict:
    import psutil
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "cpu_rss_bytes": psutil.Process().memory_info().rss,
        "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def register_logger():
    from spacy.training.loggers import console_logger
    from spacy.util import registry

    if "kurdish_nlp.ColabCheckpointLogger.v1" in registry.loggers.get_all():
        return

    @registry.loggers("kurdish_nlp.ColabCheckpointLogger.v1")
    def factory(run_path: str, drive_path: str):
        def setup(nlp, stdout=sys.stdout, stderr=sys.stderr):
            run, drive = Path(run_path), Path(drive_path)
            log, finish = console_logger(
                progress_bar=False, output_file=run / "metrics.jsonl"
            )(nlp, stdout, stderr)

            def step(info):
                log(info)
                if info is not None:
                    losses = info.get("losses", {})
                    if not all(math.isfinite(float(v)) for v in losses.values()):
                        raise RuntimeError("Non-finite loss; full training stopped")
                    cuda_model(nlp)
                    state = json.loads((run / "run.json").read_text())
                    state["last_completed_eval_step"] = info["step"]
                    state["last_gpu_memory"] = memory()
                    atomic_json(run / "run.json", state)
                    # spaCy calls its logger AFTER both checkpoint saves return.
                    # Synchronous backup prevents a concurrent partial-save race.
                    backup_run(
                        run, drive, sorted(nlp.get_pipe("ner").labels), info["step"]
                    )

            return step, finish

        return setup


def install_microbatch_guard():
    """Fix pinned spaCy's small-batch accumulation edge without losing records.

    The original function uses len(batch)//accumulation, which is zero when
    accumulation exceeds batch length, then yields the whole batch as remainder.
    Bound every yielded chunk, including the large remainder in uneven batches.
    Preserve all records and one optimizer update per effective batch.
    """
    from spacy.training import loop

    if getattr(loop.subdivide_batch, "_ner_v2_bounded", False):
        return
    original = loop.subdivide_batch

    def bounded(batch, accumulate_gradient):
        batch = list(batch)
        if batch:
            limit = max(1, len(batch) // accumulate_gradient)
            for group in original(batch, min(accumulate_gradient, len(batch))):
                for start in range(0, len(group), limit):
                    yield group[start : start + limit]

    bounded._ner_v2_bounded = True
    loop.subdivide_batch = bounded


def prepare(args):
    import spacy
    from spacy.util import load_config

    environment = gpu_environment()
    if not re.fullmatch(r"sorani-ner-v2-[A-Za-z0-9_-]+", args.run.name):
        raise ValueError("Use a descriptive new run ID beginning sorani-ner-v2-")
    if args.run.exists() or args.drive.exists():
        raise FileExistsError(
            "Run IDs are immutable; use a new ID, including for restarts"
        )
    if sha256(args.lock) != LOCK_SHA256:
        raise ValueError("Official dataset lock changed")
    lock = verify(args.data, args.lock, args.source)
    encoder = validate_encoder(args.encoder)
    base = load_config("configs/ner/xlmr-v2.cfg")
    warm = getattr(args, "snapshot", None)
    stateful = bool(warm or getattr(args, "resumable", False))
    source_model = source_marker = None
    if warm:
        from .continued import verified_source

        for output in (args.run, args.drive):
            if output.resolve().is_relative_to(
                warm.resolve()
            ) or warm.resolve().is_relative_to(output.resolve()):
                raise ValueError(
                    "Output must not overlap the immutable source snapshot"
                )
        source_model, source_marker = verified_source(
            warm, args.role, run_id=args.original_run, generation=args.generation
        )
        source_state = json.loads((warm / "run.json").read_text())
        if source_state["dataset_lock_sha256"] != sha256(args.lock):
            raise ValueError("Warm-start source used a different dataset")
        base = load_config(source_model / "config.cfg")
        source_policy = json.loads((warm / "runtime-policy.json").read_text())
        if (
            source_policy.get("gradient_checkpointing")
            and not args.gradient_checkpointing
        ):
            raise ValueError("Preserve source gradient checkpointing")
        if (
            base["components"]["transformer"]["model"]["mixed_precision"]
            != args.mixed_precision
        ):
            raise ValueError("Preserve the source checkpoint's FP16 mode")
    args.run.mkdir(parents=True)
    cfg = base.copy()
    # Preserve the serialized component architecture. Optimizer/schedule and
    # operational differences are recorded separately from the source config.
    cfg["system"]["gpu_allocator"] = "pytorch"
    cfg["training"]["gpu_allocator"] = "pytorch"
    cfg["training"]["accumulate_gradient"] = args.accumulate
    cfg["nlp"]["batch_size"] = args.eval_batch
    cfg["components"]["transformer"]["model"]["mixed_precision"] = args.mixed_precision
    if args.mixed_precision:
        cfg["components"]["transformer"]["model"]["grad_scaler_config"] = {
            "init_scale": 128.0
        }
    cfg["components"]["transformer"]["model"]["name"] = str(args.encoder.resolve())
    cfg["training"]["logger"] = {
        "@loggers": "kurdish_nlp.ColabCheckpointLogger.v1",
        "run_path": str(args.run.resolve()),
        "drive_path": str(args.drive.resolve()),
    }
    for split in ("train", "dev"):
        records = read_jsonl(args.data / f"{split}.jsonl")
        path = args.run / f"{split}.spacy"
        convert_records(records, path, spacy.blank("xx"))
        cfg["paths"][split] = str(path.resolve())
    policy = {
        "version": 1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_overflow_retries": 8,
        "parameter_unscale": "once_per_effective_batch",
        "overflow": "discard_both_components_and_replay_same_batch",
        "windows": "reject_truncation_and_verify_all_window_counts",
    }
    if stateful:
        policy.update(
            state_format=2,
            checkpoint_frequency=getattr(args, "checkpoint_frequency", 200),
            schedule={
                "peak": getattr(args, "learning_rate", 5e-6) if warm else 2e-5,
                "warmup": getattr(args, "warmup", 200),
                "total": getattr(args, "updates", 2000) if warm else 20000,
            },
        )
    from .training_state import learning_rate

    if stateful:
        learning_rate(policy["schedule"], 0)
        cfg["training"]["max_steps"] = policy["schedule"]["total"]
        cfg["training"]["optimizer"]["learn_rate"] = learning_rate(
            policy["schedule"], 0
        )
    if warm:
        cfg["training"]["patience"] = args.patience
        cfg["training"]["eval_frequency"] = args.eval_frequency
    dump(args.run / "runtime-policy.json", policy)
    cfg.to_disk(args.run / "config.cfg")
    changes = {}

    def differences(a, b, prefix=""):
        for key in sorted(set(a) | set(b)):
            name = f"{prefix}.{key}".strip(".")
            x, y = a.get(key), b.get(key)
            if isinstance(x, dict) and isinstance(y, dict):
                differences(x, y, name)
            elif x != y:
                changes[name] = {"base": x, "effective": y}

    differences(base, cfg)
    if stateful:
        changes["runtime.training_loop"] = {
            "base": "spaCy zero-based steps; opaque epoch iterator; weights-only saves",
            "effective": "completed update counts; persisted padded batch plan; positive first warmup update; full-state generations; strict dev F1 improvement; dev evaluation every eval_frequency completed updates",
        }
    changes["runtime.microbatch_subdivision"] = {
        "base": "spaCy 3.8.7 can leave a small batch unsplit when accumulation exceeds record count",
        "effective": "bound every microbatch including uneven remainders to max(1, records // accumulation); effective batch, optimizer update count and schedule unchanged",
    }
    changes["runtime.numerics"] = {
        "base": "per-backward parameter unscale",
        "effective": policy,
    }
    dump(args.run / "effective-changes.json", changes)
    dump(args.run / "environment.json", environment)
    (args.run / "dependencies.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    code = json.loads(Path("code-manifest.json").read_text())
    validate_code_manifest(code)
    dump(args.run / "code-manifest.json", code)
    state = {
        "run_id": args.run.name,
        "kind": "full",
        "status": "prepared",
        "dataset_lock_sha256": sha256(args.lock),
        "encoder": encoder,
        "config_sha256": sha256(args.run / "config.cfg"),
        "runtime_policy_sha256": sha256(args.run / "runtime-policy.json"),
        "base_config_sha256": sha256(Path("configs/ner/xlmr-v2.cfg")),
        "corpora_sha256": {
            s: sha256(args.run / f"{s}.spacy") for s in ("train", "dev")
        },
        "git_commit": code["base_git_commit"],
        "working_tree_content_sha256": code["content_sha256"],
        "labels": lock["labels"],
        "test_predictions_made": False,
        "exact_resume_supported": stateful,
        "numerical_reproducibility": "full state recovery; CUDA bitwise equality is not guaranteed"
        if stateful
        else "weights-only; new-run restart required",
        "state_drive": str((args.drive / "training-state").resolve())
        if stateful
        else None,
        "restart_policy": "resume verified full state only; weights-only requires a new warm-start run"
        if stateful
        else "new run, pinned pretrained encoder, fresh optimizer and scheduler",
        "microbatch_workaround": "spaCy 3.8.7 microbatches bounded to max(1, records // accumulation), including small batches and uneven remainders; effective batch/optimizer schedule unchanged",
    }
    if warm:
        from .training_state import weight_fingerprints

        # These are private local copies. The verified original is only read.
        shutil.copytree(source_model, args.run / "initial-model")
        shutil.copytree(warm / "checkpoints/model-best", args.run / "baseline-model")
        if hashes(args.run / "initial-model") != hashes(source_model) or hashes(
            args.run / "baseline-model"
        ) != hashes(warm / "checkpoints/model-best"):
            raise ValueError("Incomplete warm-start checkpoint copy")
        best_meta = json.loads((warm / "checkpoints/model-best/meta.json").read_text())
        observed = best_meta.get("performance", {}).get("ents_f")
        if observed is None or not math.isclose(
            float(observed) / 100, args.baseline_f1, abs_tol=0.00005
        ):
            raise ValueError(
                "Source model-best metadata does not match declared baseline F1"
            )
        model = spacy.load(args.run / "initial-model")
        if sorted(model.get_pipe("ner").labels) != lock["labels"]:
            raise ValueError("Warm-start labels differ from frozen release")
        state["warm_start"] = {
            "original_run": source_marker["run_id"],
            "checkpoint_generation": args.generation,
            "source_step": source_marker.get("step", source_marker.get("updates")),
            "source_resume_state": source_marker["resume_state"],
            "source_config_sha256": sha256(source_model / "config.cfg"),
            "source_complete_sha256": sha256(warm / "COMPLETE.json"),
            "source_model_role": args.role,
            "source_model_files": hashes(source_model),
            "initial_weights": weight_fingerprints(model),
            "strategy": "spacy.load; preserve components/tokenizer/vocab/moves; fresh Adam and positive warmup-linear schedule",
            "optimizer_restored": False,
            "baseline_model_files": hashes(warm / "checkpoints/model-best"),
        }
        state["baseline_f1"] = args.baseline_f1
        state["baseline_exact_f1"] = float(observed) / 100
        dump(
            args.run / "baseline.json",
            {
                "dev_entity_f1": args.baseline_f1,
                "dev_entity_f1_unrounded": state["baseline_exact_f1"],
                "original_best_step": 5400
                if source_marker["run_id"] == "sorani-ner-v2-t4-20261022-104347"
                else source_state.get("best_update"),
                "source": state["warm_start"],
                "test_predictions_made": False,
            },
        )
        del model
        gc.collect()
    atomic_json(args.run / "run.json", state)
    print(
        json.dumps(
            {"run": str(args.run), "changes": changes, "environment": environment},
            indent=2,
        )
    )


def load_run(args):
    from spacy.util import load_config

    environment = gpu_environment()
    state = json.loads((args.run / "run.json").read_text())
    code = json.loads((args.run / "code-manifest.json").read_text())
    validate_code_manifest(code)
    if code["content_sha256"] != state["working_tree_content_sha256"]:
        raise ValueError(
            "Source checkout changed since preparation; create a new verified run"
        )
    if (
        state["config_sha256"] != sha256(args.run / "config.cfg")
        or state.get("runtime_policy_sha256")
        != sha256(args.run / "runtime-policy.json")
        or sha256(args.lock) != state["dataset_lock_sha256"]
    ):
        raise ValueError(
            "Run configuration or dataset identity changed; create a new run"
        )
    verify(args.data, args.lock, args.source)
    for split, digest in state["corpora_sha256"].items():
        if not (args.run / f"{split}.spacy").exists() and (
            state["status"] == "trained" or state.get("exact_resume_supported")
        ):
            # Backup generations intentionally omit corpus text. Rehydrate only
            # from the already verified release, then require identical bytes.
            import spacy

            convert_records(
                read_jsonl(args.data / f"{split}.jsonl"),
                args.run / f"{split}.spacy",
                spacy.blank("xx"),
            )
        if sha256(args.run / f"{split}.spacy") != digest:
            raise ValueError("Run corpus changed")
    cfg = load_config(args.run / "config.cfg")
    if not state.get("warm_start") and getattr(args, "command", "") != "resume":
        validate_encoder(Path(cfg["components"]["transformer"]["model"]["name"]))
    register_logger()
    install_microbatch_guard()
    install_window_guard()
    return state, cfg, environment


def _examples(nlp, records):
    from spacy.training import Example

    return [
        Example(nlp.make_doc(" ".join(r["tokens"])), gold_doc(nlp, r)) for r in records
    ]


def _update(nlp, batch, optimizer, cfg):
    cuda_model(nlp)
    if "schedule" in nlp._ner_v2_policy:
        from .training_state import learning_rate

        optimizer.learn_rate = learning_rate(
            nlp._ner_v2_policy["schedule"], getattr(nlp, "_ner_v2_update_index", 0)
        )
    result = checked_update(
        nlp,
        batch,
        optimizer,
        accumulate=cfg["training"]["accumulate_gradient"],
        dropout=cfg["training"]["dropout"],
        policy=nlp._ner_v2_policy,
        diagnostic_path=nlp._ner_v2_diagnostics,
    )
    return {**result, **memory()}


def runtime_model(nlp, args):
    policy = json.loads((args.run / "runtime-policy.json").read_text())
    nlp._ner_v2_stage = getattr(args, "stage", "training")
    return configure_model(nlp, policy, args.run / "numerics.jsonl")


def preflight(args):
    import spacy
    import torch
    from spacy.tokens import DocBin
    from spacy.util import registry

    state, cfg, environment = load_run(args)
    output = args.run / f"preflight-{args.stage}.json"
    if output.exists():
        raise FileExistsError(
            "Preflight results are immutable; create a new run to change memory settings"
        )
    if args.stage != "smoke" and not (args.run / "preflight-smoke.json").exists():
        raise ValueError("Complete CUDA smoke first")
    train_records = read_jsonl(args.data / "train.jsonl")
    dev_records = read_jsonl(args.data / "dev.jsonl")
    report = {
        "stage": args.stage,
        "config_sha256": state["config_sha256"],
        "runtime_policy_sha256": state["runtime_policy_sha256"],
        "environment": environment,
        "test_predictions_made": False,
    }
    try:
        if args.stage == "smoke":
            # Init all real labels from the actual train DocBin, without any test data.
            from .continued import load_initial

            nlp = load_initial(args.run, cfg, state)
            report["runtime"] = runtime_model(nlp, args)
            if sorted(nlp.get_pipe("ner").labels) != state["labels"]:
                raise ValueError("Initialization dropped a legitimate entity type")
            hf = cuda_model(nlp)
            optimizer = nlp.create_optimizer()
            selected = []
            for label in state["labels"]:
                record = min(
                    (
                        r
                        for r in train_records
                        if any(s[2] == label for s in spans(r["tags"]))
                    ),
                    key=lambda r: (len(r["tokens"]), r["id"]),
                )
                if record not in selected:
                    selected.append(record)
            batch = _examples(nlp, selected[:4])
            weight = hf.encoder.layer[0].attention.self.query.weight
            before = weight.detach().clone()
            from .training_state import fresh_optimizer, weight_fingerprints

            initial = weight_fingerprints(nlp)
            if state.get("exact_resume_supported"):
                optimizer = fresh_optimizer(nlp, nlp._ner_v2_policy["schedule"])
            torch.cuda.reset_peak_memory_stats()
            updates = [_update(nlp, batch, optimizer, cfg) for _ in range(3)]
            after = weight_fingerprints(nlp)
            if any(initial[k] == after[k] for k in ("transformer", "ner")):
                raise RuntimeError(
                    "Smoke did not update both transformer and NER weights"
                )
            if torch.equal(before, weight.detach()):
                raise RuntimeError(
                    "CUDA optimizer did not change transformer parameters"
                )
            text = "😀 " + " ".join(dev_records[0]["tokens"])
            doc = nlp(text)
            payload = entities_payload(
                doc, "sorani-ner-v2-smoke", "0.0.0-smoke", state["labels"]
            )
            if not any(
                t.__class__.__module__.startswith("cupy")
                for t in doc._.trf_data.tensors
            ):
                raise RuntimeError("Transformer output is not CUDA/CuPy")
            model = args.run / "cuda-smoke-model"
            nlp.to_disk(model)
            validate_pipeline_files(model, state["labels"])
            size = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
            del nlp, optimizer, hf, weight, before, batch, doc
            gc.collect()
            torch.cuda.empty_cache()
            spacy.require_gpu(0)
            loaded = spacy.load(model)
            runtime_model(loaded, args)
            cuda_model(loaded)
            again = entities_payload(
                loaded(text), "sorani-ner-v2-smoke", "0.0.0-smoke", state["labels"]
            )
            if again != payload:
                raise ValueError("Checkpoint reload changed predictions")
            report.update(
                updates=updates,
                transformer_weight_changed=True,
                initial_weights=initial,
                updated_weights=after,
                source_weights_equal=initial
                == state.get("warm_start", {}).get("initial_weights")
                if state.get("warm_start")
                else None,
                reload_predictions_equal=True,
                prediction_example=payload,
                checkpoint_bytes=size,
                checkpoints_active_bytes=2 * size,
                drive_two_generations_bytes=4 * size,
                full_state_estimated_generation_bytes=4 * size,
                full_state_retention="all verified generations retained; provision disk and Drive capacity",
            )
        else:
            spacy.require_gpu(0)
            nlp = spacy.load(args.run / "cuda-smoke-model")
            report["runtime"] = runtime_model(nlp, args)
            cuda_model(nlp)
            optimizer = nlp.create_optimizer()
            resolved = registry.resolve({"batcher": cfg["training"]["batcher"]})[
                "batcher"
            ]
            if args.stage == "stress":
                # Full worst-case singleton is never subdivided by accumulation.
                longest = sorted(
                    train_records, key=lambda r: (-len(r["tokens"]), r["id"])
                )[:8]
                examples = _examples(nlp, longest)
                batches = list(resolved(examples))
                torch.cuda.reset_peak_memory_stats()
                updates = [_update(nlp, batch, optimizer, cfg) for batch in batches]
                # The source's 1,992-token record belongs to TEST. Never read it
                # for memory tuning. Mechanically repeat complete TRAIN annotations
                # in memory only, exceeding that shape without inventing labels.
                # This engineering fixture is never exported into the dataset or
                # used to update the production model.
                record = longest[0]
                repeats = math.ceil(1992 / len(record["tokens"]))
                envelope = {
                    **record,
                    "id": "engineering-shape-only",
                    "tokens": record["tokens"] * repeats,
                    "tags": record["tags"] * repeats,
                }
                envelope_update = _update(
                    nlp, _examples(nlp, [envelope]), optimizer, cfg
                )
                # Validate the actual inference/evaluation batch size on longest dev inputs.
                texts = [
                    " ".join(r["tokens"])
                    for r in sorted(dev_records, key=lambda r: -len(r["tokens"]))[
                        : cfg["nlp"]["batch_size"]
                    ]
                ]
                for doc in nlp.pipe(texts, batch_size=cfg["nlp"]["batch_size"]):
                    entities_payload(
                        doc, "sorani-ner-v2-smoke", "0.0.0-smoke", state["labels"]
                    )
                report.update(
                    records=[
                        {"id": r["id"], "bio_tokens": len(r["tokens"])} for r in longest
                    ],
                    updates=updates,
                    memory=memory(),
                    eval_batch_size=cfg["nlp"]["batch_size"],
                    shape_envelope={
                        "training_source": record["id"],
                        "tokens": len(envelope["tokens"]),
                        "complete_repeats": repeats,
                        "update": envelope_update,
                        "note": "Engineering-only repeated TRAIN annotations; no test content, new human labels, or dataset additions. Conservative shape exceeds the source maximum 1992.",
                    },
                )
            else:
                if not (args.run / "preflight-stress.json").exists():
                    raise ValueError("Complete GPU memory stress test first")
                records = list(train_records)
                random.Random(1729).shuffle(records)
                sample = _examples(nlp, records[:256])
                batches = list(resolved(sample))
                torch.cuda.reset_peak_memory_stats()
                timings = [
                    _update(nlp, batch, optimizer, cfg) for batch in batches[:12]
                ]
                # The first two updates include optimizer warmup/allocation overhead.
                measured = timings[2:]
                seconds = sum(x["seconds"] for x in measured)
                if not measured or seconds <= 0:
                    raise RuntimeError("Insufficient GPU timing samples")
                all_docs = list(
                    DocBin().from_disk(args.run / "train.spacy").get_docs(nlp.vocab)
                )
                # Count one actual epoch's batches under the same padded batcher.
                epoch_batches = sum(
                    1
                    for _ in resolved(
                        [
                            __import__("spacy").training.Example(d.copy(), d)
                            for d in all_docs
                        ]
                    )
                )
                start = time.perf_counter()
                refs = _examples(nlp, dev_records)
                nlp.evaluate(refs, batch_size=cfg["nlp"]["batch_size"])
                torch.cuda.synchronize()
                dev_seconds = time.perf_counter() - start
                seconds_per_update = seconds / len(measured)
                max_updates = min(
                    epoch_batches * cfg["training"]["max_epochs"],
                    cfg["training"]["max_steps"]
                    + (0 if state.get("exact_resume_supported") else 1),
                )
                report.update(
                    updates=timings,
                    measured_updates=len(measured),
                    tokens_per_second=sum(x["tokens"] for x in measured) / seconds,
                    updates_per_second=len(measured) / seconds,
                    batches_per_epoch=epoch_batches,
                    estimated_epoch_seconds=epoch_batches * seconds_per_update,
                    full_dev_evaluation_seconds=dev_seconds,
                    estimated_full_seconds=max_updates * seconds_per_update
                    + (max_updates // cfg["training"]["eval_frequency"] + 1)
                    * dev_seconds,
                    max_updates=max_updates,
                    memory=memory(),
                    estimate_note="Measured GPU training and full-dev evaluation only; Drive backup/checkpoint writes and early stopping change duration.",
                )
        report["status"] = "passed"
        dump(output, report)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            diagnostics=error.diagnostics
            if isinstance(error, NumericalFailure)
            else None,
            recovery="Create a NEW run ID. Increase accumulation to subdivide multi-record batches; keep padded size 256. A singleton long segment cannot be subdivided: enable FP16 plus gradient checkpointing and repeat all preflights, or use more VRAM. Inspect numerics.jsonl for named gradients/scaler attempts. Never drop/truncate records. Reduce eval batch size if inference OOMs.",
        )
        try:
            report["failure_memory"] = memory()
        except (RuntimeError, OSError) as memory_error:
            report["memory_error"] = repr(memory_error)
        dump(output, report)
        raise


def gate(args) -> dict:
    state, cfg, _ = load_run(args)
    for stage in ("smoke", "stress", "benchmark"):
        report = json.loads((args.run / f"preflight-{stage}.json").read_text())
        if (
            report["status"] != "passed"
            or report["config_sha256"] != state["config_sha256"]
            or report.get("runtime_policy_sha256") != state.get("runtime_policy_sha256")
            or not report.get("updates")
            or any(u.get("optimizer_updates") != 1 for u in report["updates"])
            or (
                state.get("warm_start")
                and stage == "smoke"
                and not report.get("source_weights_equal")
            )
        ):
            raise ValueError(
                "All three preflights must pass on the unchanged effective configuration"
            )
    if cfg["components"]["transformer"]["model"]["mixed_precision"]:
        print(
            "FP16 explicitly enabled; finite CUDA gradients, changed weights and stress/benchmark updates validated."
        )
    return state


def full_train(args):
    import spacy
    from spacy.training.initialize import init_nlp
    from spacy.training.loop import train

    state = gate(args)
    if state["status"] != "prepared" or (args.run / "checkpoints").exists():
        raise FileExistsError(
            "No pseudo-resume: full training starts only in a fresh prepared run"
        )
    _, cfg, _ = load_run(args)
    # Must follow anti-resume validation and precede every spaCy checkpoint write.
    (args.run / "checkpoints").mkdir()
    if state.get("exact_resume_supported"):
        from .continued import train_run

        _window_preflight(args, state, cfg)
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        return train_run(args, state, cfg)
    # All train/dev wordpiece windows must fit; no transformer truncation accepted.
    encoder = Path(cfg["components"]["transformer"]["model"]["name"])
    wp = wordpiece_check(
        [r for s in ("train", "dev") for r in read_jsonl(args.data / f"{s}.jsonl")],
        spacy.blank("xx"),
        encoder,
        32,
        24,
    )
    state.update(status="running", wordpiece_preflight=wp)
    atomic_json(args.run / "run.json", state)
    try:
        nlp = init_nlp(cfg, use_gpu=0)
        runtime_model(nlp, args)
        install_training_loop(nlp._ner_v2_policy, args.run / "numerics.jsonl")
        cuda_model(nlp)
        nlp.meta.update(
            name="sorani_ner_v2",
            version="2.0.0",
            description="Sorani NER V2; original article identities unavailable",
        )
        train(nlp, args.run / "checkpoints", use_gpu=0)
        state = json.loads((args.run / "run.json").read_text())
        state["status"] = "trained"
        atomic_json(args.run / "run.json", state)
        backup_run(
            args.run,
            Path(cfg["training"]["logger"]["drive_path"]),
            state["labels"],
            state["last_completed_eval_step"],
        )
    except BaseException as error:
        state = json.loads((args.run / "run.json").read_text())
        state.update(status="interrupted_or_failed", error_type=type(error).__name__)
        atomic_json(args.run / "run.json", state)
        raise


def _window_preflight(args, state, cfg):
    import spacy

    from .continued import load_initial

    # Warm-start checks use the saved tokenizer, not a fresh blank tokenizer.
    nlp = (
        load_initial(args.run, cfg, state)
        if state.get("warm_start")
        else spacy.blank("xx")
    )
    window = cfg["components"]["transformer"]["model"]["get_spans"]
    tokenizer = (
        nlp.get_pipe("transformer").model.tokenizer if state.get("warm_start") else None
    )
    state["wordpiece_preflight"] = wordpiece_check(
        [r for s in ("train", "dev") for r in read_jsonl(args.data / f"{s}.jsonl")],
        nlp,
        Path(cfg["components"]["transformer"]["model"]["name"]),
        window["window"],
        window["stride"],
        tokenizer=tokenizer,
    )
    atomic_json(args.run / "run.json", state)


def resume(args):
    from .continued import train_run

    state = gate(args)
    if not state.get("exact_resume_supported") or state["status"] not in {
        "running",
        "interrupted_or_failed",
    }:
        raise ValueError(
            "True resume requires an interrupted full-state run; no weights-only fallback"
        )
    _, cfg, _ = load_run(args)
    (args.run / "checkpoints").mkdir(exist_ok=True)
    return train_run(args, state, cfg, resume=True)


def compare_dev(args):
    """Development-only comparison; no test-set selection or tuning."""
    import spacy

    from .continued import identity
    from .training_state import latest_generation, validate_generation

    state, cfg, _ = load_run(args)
    generation = (
        latest_generation(args.run / "state-checkpoints")
        if (args.run / "state-checkpoints/latest.json").exists()
        else args.run
    )
    validate_generation(generation, identity(state))
    nlp = spacy.load(generation / "checkpoints/model-best")
    runtime_model(nlp, args)
    scores = nlp.evaluate(
        _examples(nlp, read_jsonl(args.data / "dev.jsonl")),
        batch_size=cfg["nlp"]["batch_size"],
    )
    result = {
        "run_id": state["run_id"],
        "baseline_f1": state.get("baseline_f1"),
        "baseline_exact_f1": state.get("baseline_exact_f1"),
        "selected_dev_f1": scores["ents_f"] / 100,
        "test_predictions_made": False,
        "model_files": hashes(generation / "checkpoints/model-best"),
    }
    if state.get("baseline_f1") is not None:
        result["delta_f1"] = result["selected_dev_f1"] - state["baseline_f1"]
        result["delta_f1_unrounded"] = result["selected_dev_f1"] - state.get(
            "baseline_exact_f1", state["baseline_f1"]
        )
    print(json.dumps(result, indent=2))


def export_model(args):
    import spacy

    from .training import tree_hash

    state, _, _ = load_run(args)
    selection = json.loads((args.run / "selection.lock.json").read_text())
    evaluation = json.loads((args.run / "test-evaluation.json").read_text())
    receipt = json.loads((args.run / "test-evaluation.receipt.json").read_text())
    model = args.run / "checkpoints/model-best"
    if state["status"] != "trained" or selection["model_files"] != tree_hash(model):
        raise ValueError("Only a frozen completed full model may be exported")
    if evaluation["selection_sha256"] != sha256(
        args.run / "selection.lock.json"
    ) or receipt["report_sha256"] != sha256(args.run / "test-evaluation.json"):
        raise ValueError("Missing or changed frozen evaluation")
    if args.output.exists():
        raise FileExistsError(args.output)
    # CPU deployment load is intentionally checked in this fresh export process.
    spacy.require_cpu()
    nlp = spacy.load(model)
    if sorted(nlp.get_pipe("ner").labels) != state["labels"]:
        raise ValueError("Export label inventory mismatch")
    for text in ("ئاژانسی ناسا پلانی هەیە", "😀 سڵاو لە هەولێر"):
        entities_payload(nlp(text), "sorani-ner-v2", "2.0.0", state["labels"])
    import tarfile

    files = {f"sorani-ner-v2/model/{name}": model / name for name in hashes(model)}
    for name in (
        "config.cfg",
        "run.json",
        "dependencies.txt",
        "environment.json",
        "code-manifest.json",
        "effective-changes.json",
        "runtime-policy.json",
        "numerics.jsonl",
        "selection.lock.json",
        "test-evaluation.json",
        "test-evaluation.receipt.json",
        "metrics.jsonl",
        "train.log",
        "preflight-smoke.json",
        "preflight-stress.json",
        "preflight-benchmark.json",
    ):
        if (args.run / name).is_file():
            files[f"sorani-ner-v2/{name}"] = args.run / name
    files["sorani-ner-v2/dataset-release.lock.json"] = args.lock
    manifest = {
        "model_id": "sorani-ner-v2",
        "version": "2.0.0",
        "run_id": state["run_id"],
        "dataset_lock_sha256": sha256(args.lock),
        "cpu_load_and_offsets_validated": True,
        "files": {name: sha256(path) for name, path in sorted(files.items())},
        "privacy": "Private model artifact; dataset redistribution license unverified. No raw data included.",
    }
    import io

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(
            "An incomplete prior export exists; use a new output name"
        )
    with tarfile.open(temporary, "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("sorani-ner-v2/integrity.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(temporary) as archive:
        for name, digest in manifest["files"].items():
            import hashlib

            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError("Export archive integrity failure")
    os.replace(temporary, args.output)
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".sha256.json"),
        {"sha256": sha256(args.output), **manifest},
    )
    print(
        json.dumps(
            {
                "export": str(args.output),
                "sha256": sha256(args.output),
                "model_id": "sorani-ner-v2",
            },
            indent=2,
        )
    )


def main():
    from .cuda_runtime import bootstrap

    bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "environment",
        "prepare",
        "prepare-continued",
        "preflight",
        "gate",
        "train",
        "resume",
        "compare-dev",
        "export",
    ):
        p = commands.add_parser(name)
        p.add_argument("--run", type=Path)
        p.add_argument("--source", type=Path, default=Path("data/ner/raw/adyan.txt"))
        p.add_argument(
            "--data", type=Path, default=Path("data/ner/builds/sorani-ner-v2.0")
        )
        p.add_argument(
            "--lock", type=Path, default=Path("configs/ner/ner-v2.lock.json")
        )
        if name in {"prepare", "prepare-continued"}:
            p.add_argument("--encoder", type=Path, required=True)
            p.add_argument("--drive", type=Path, required=True)
            p.add_argument("--accumulate", type=int, default=4, choices=(4, 8, 16, 32))
            p.add_argument(
                "--eval-batch", type=int, default=16, choices=(1, 2, 4, 8, 16)
            )
            p.add_argument("--mixed-precision", action="store_true")
            p.add_argument("--gradient-checkpointing", action="store_true")
            p.add_argument("--checkpoint-frequency", type=int, default=200)
        if name == "prepare":
            p.add_argument(
                "--resumable",
                action="store_true",
                help="Opt in to full training-state checkpoints; default preserves the original notebook workflow",
            )
        if name == "prepare-continued":
            p.add_argument("--snapshot", type=Path, required=True)
            p.add_argument(
                "--role", choices=("model-last", "model-best"), default="model-last"
            )
            p.add_argument("--original-run", required=True)
            p.add_argument("--generation", required=True)
            p.add_argument("--baseline-f1", type=float, default=0.8733)
            p.add_argument("--learning-rate", type=float, default=5e-6)
            p.add_argument("--warmup", type=int, default=200)
            p.add_argument("--updates", type=int, default=2000)
            p.add_argument("--patience", type=int, default=600)
            p.add_argument("--eval-frequency", type=int, default=200)
        if name == "preflight":
            p.add_argument(
                "--stage", choices=("smoke", "stress", "benchmark"), required=True
            )
        if name == "export":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "environment" and args.run is None:
        parser.error("--run is required")
    for name in ("checkpoint_frequency", "updates", "patience", "eval_frequency"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "baseline_f1") and not 0 <= args.baseline_f1 <= 1:
        parser.error("--baseline-f1 must be in [0, 1]")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    if args.command == "environment":
        print(json.dumps(gpu_environment(), indent=2))
    elif args.command == "train":
        full_train(args)
    elif args.command == "export":
        export_model(args)
    else:
        if args.run is None:
            parser.error("--run is required")
        action = {"prepare-continued": prepare, "compare-dev": compare_dev}.get(
            args.command
        )
        result = (action or globals()[args.command])(args)
        if args.command == "gate":
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

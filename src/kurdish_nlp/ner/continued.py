"""Warm-start loading and stateful effective-batch training, without initialization."""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

from .colab_artifacts import atomic_json, hashes, validate_snapshot
from .colab_numerics import NumericalFailure, checked_update, configure_model
from .training_state import (
    fresh_optimizer,
    learning_rate,
    load_state,
    publish_generation,
    replicate_generation,
    save_state,
    validate_generation,
    weight_fingerprints,
)


def verified_source(path, role, *, run_id=None, generation=None):
    marker = validate_snapshot(path)
    if run_id is not None and marker["run_id"] != run_id:
        raise ValueError("Source run differs from requested original run")
    if generation is not None and path.name != generation:
        raise ValueError("Source generation differs from requested generation")
    if marker.get("resume_state") == "full_training_state":
        validate_generation(path)
    elif marker.get("resume_state") != "weights_only_no_optimizer_or_scheduler":
        raise ValueError("Warm-start requires a declared checkpoint state format")
    return path / "checkpoints" / role, marker


def load_initial(run, cfg, state):
    import spacy
    from spacy.training.initialize import init_nlp

    if state.get("warm_start"):
        path = run / "initial-model"
        if hashes(path) != state["warm_start"]["source_model_files"]:
            raise ValueError("Warm-start checkpoint changed after verification")
        nlp = spacy.load(path)  # NEVER initialize() / init_nlp() / resume_training().
        before = weight_fingerprints(nlp)
        if before != state["warm_start"]["initial_weights"]:
            raise ValueError("Initial weights differ from verified source checkpoint")
        # Keep architecture, tokenizer, vocabulary, NER moves and label IDs loaded
        # from disk. Only orchestration sections may change.
        for section in ("training", "paths", "system"):
            nlp.config[section] = cfg[section]
        if weight_fingerprints(nlp) != before:
            raise ValueError("Warm-start configuration altered trained weights")
    else:
        nlp = init_nlp(cfg, use_gpu=0)
    return nlp


def identity(state):
    value = {
        k: state[k]
        for k in (
            "run_id",
            "dataset_lock_sha256",
            "config_sha256",
            "runtime_policy_sha256",
            "corpora_sha256",
            "working_tree_content_sha256",
        )
    }
    value.update(
        {k: state.get(k) for k in ("warm_start", "baseline_f1", "baseline_exact_f1")}
    )
    return value


def batch_plan(nlp, records, cfg, epoch):
    """Persist the actual padded-batcher order, independent of model RNG."""
    from spacy.util import registry

    from .colab import _examples

    examples = _examples(nlp, records)
    random.Random(cfg["training"]["seed"] + epoch).shuffle(examples)
    batcher = registry.resolve({"batcher": cfg["training"]["batcher"]})["batcher"]
    plan = [
        [e.reference.user_data["record_id"] for e in batch]
        for batch in batcher(examples)
    ]
    flat = [key for batch in plan for key in batch]
    expected = [r["id"] for r in records]
    if not plan or sorted(flat) != sorted(expected) or len(set(flat)) != len(flat):
        raise ValueError("Batch plan dropped or duplicated training records")
    return plan


def validate_progress(progress, records, schedule):
    flat = [key for batch in progress["plan"] for key in batch]
    if (
        sorted(flat) != sorted(r["id"] for r in records)
        or any(not batch for batch in progress["plan"])
        or not 0 <= progress["cursor"] <= len(progress["plan"])
        or not 0 <= progress["updates"] <= schedule["total"]
        or progress["epoch"] < 0
        or progress["best_update"] > progress["updates"]
        or not math.isfinite(progress["best_f1"])
    ):
        raise ValueError("Invalid sampler/update/selection state")


def fit(
    nlp,
    optimizer,
    records,
    dev_records,
    cfg,
    policy,
    progress,
    *,
    evaluate,
    checkpoint,
    diagnostic_path=None,
    device="cuda",
    stop_after=None,
):
    """Shared production/test loop. Checkpoint callbacks see only clean boundaries.

    A process failure within update/evaluation/save cannot advance a published
    cursor. Resuming repeats all work after the last verified generation.
    """
    from .colab import _examples

    schedule = policy["schedule"]
    validate_progress(progress, records, schedule)
    by_id = {r["id"]: r for r in records}
    max_epochs = cfg["training"]["max_epochs"]
    frequency = cfg["training"]["eval_frequency"]
    patience = cfg["training"]["patience"]
    while progress["updates"] < schedule["total"] and not progress["finished"]:
        if progress["cursor"] == len(progress["plan"]):
            progress["epoch"] += 1
            if max_epochs > 0 and progress["epoch"] >= max_epochs:
                progress["finished"] = True
                break
            progress["plan"] = batch_plan(nlp, records, cfg, progress["epoch"])
            progress["cursor"] = 0
        keys = progress["plan"][progress["cursor"]]
        batch = _examples(nlp, [by_id[key] for key in keys])
        optimizer.learn_rate = learning_rate(schedule, progress["updates"])
        update = checked_update(
            nlp,
            batch,
            optimizer,
            accumulate=cfg["training"]["accumulate_gradient"],
            dropout=cfg["training"]["dropout"],
            policy=policy,
            diagnostic_path=diagnostic_path,
            expected_device=device,
        )
        progress["updates"] += 1
        progress["cursor"] += 1
        progress["words"] += update["tokens"]
        at_end = progress["updates"] == schedule["total"] or (
            max_epochs > 0
            and progress["epoch"] + 1 == max_epochs
            and progress["cursor"] == len(progress["plan"])
        )
        scored = progress["updates"] % frequency == 0 or at_end
        improved = False
        if scored:
            scores = evaluate(nlp, dev_records)
            score = float(scores["ents_f"]) / 100.0
            if not math.isfinite(score):
                raise NumericalFailure("nonfinite_development_score", {})
            progress["last_scores"] = scores
            # Strict improvements preserve the historical baseline on ties.
            improved = score > progress["best_f1"]
            if improved:
                progress["best_f1"], progress["best_update"] = (
                    score,
                    progress["updates"],
                )
        progress["finished"] = bool(
            at_end
            or (patience and progress["updates"] - progress["best_update"] >= patience)
        )
        if (
            scored
            or progress["updates"] % policy["checkpoint_frequency"] == 0
            or progress["finished"]
        ):
            checkpoint(nlp, optimizer, progress, improved)
        if stop_after is not None and progress["updates"] >= stop_after:
            return progress
    return progress


def train_run(args, state, cfg, *, resume=False, device="cuda"):
    """One local writer per run; OS releases the lock after runtime/process loss."""
    import fcntl

    run = args.run.resolve()
    with (run.parent / f".{run.name}.trainer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another trainer is already writing this local run"
            ) from error
        try:
            return _train_run(args, state, cfg, resume=resume, device=device)
        except BaseException as error:
            state.update(
                status="interrupted_or_failed", error_type=type(error).__name__
            )
            atomic_json(args.run / "run.json", state)
            raise


def _train_run(args, state, cfg, *, resume=False, device="cuda"):
    import spacy
    from spacy.util import fix_random_seed

    from .build import read_jsonl
    from .colab import _examples, cuda_model, memory

    run = args.run
    policy = json.loads((run / "runtime-policy.json").read_text())
    if policy.get("state_format") != 2:
        raise ValueError(
            "This run has no full training-state support; never pseudo-resume"
        )
    if any(
        cfg["training"].get(k)
        for k in (
            "frozen_components",
            "annotating_components",
            "before_update",
            "before_to_disk",
        )
    ):
        raise ValueError(
            "Stateful loop supports only the audited transformer -> NER configuration without custom callbacks"
        )
    records = read_jsonl(args.data / "train.jsonl")
    dev_records = read_jsonl(args.data / "dev.jsonl")
    if resume:
        # A restored generation is at run root; local sessions use a generation
        # pointer. Never load mutable convenience model-last for true resume.
        from .training_state import latest_generation

        generation = latest_generation(run / "state-checkpoints")
        validate_generation(generation, identity(state))
        saved_run = json.loads((generation / "run.json").read_text())
        if identity(saved_run) != identity(state):
            raise ValueError("Generation run metadata differs from prepared run")
        nlp = spacy.load(generation / "checkpoints/model-last")
        for section in ("training", "paths", "system"):
            nlp.config[section] = cfg[section]
        configure_model(nlp, policy, run / "numerics.jsonl")
        optimizer = fresh_optimizer(nlp, policy["schedule"])
        progress = load_state(generation / "training-state", nlp, optimizer)
        validate_progress(progress, records, policy["schedule"])
        if progress["finished"] and state["status"] != "interrupted_or_failed":
            raise ValueError(
                "Checkpoint is terminal; prepare a NEW continued-training experiment"
            )
        if generation != run:
            target = run / "checkpoints/model-best"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                generation / "checkpoints/model-best",
                target,
            )
        if (generation / "metrics.jsonl").is_file():
            shutil.copy2(generation / "metrics.jsonl", run / "metrics.jsonl")
    else:
        fix_random_seed(cfg["training"]["seed"])
        nlp = load_initial(run, cfg, state)
        configure_model(nlp, policy, run / "numerics.jsonl")
        optimizer = fresh_optimizer(nlp, policy["schedule"])
        progress = {
            "updates": 0,
            "epoch": 0,
            "cursor": 0,
            "words": 0,
            "plan": batch_plan(nlp, records, cfg, 0),
            "finished": False,
            "best_f1": max(
                state.get("baseline_f1", -1.0), state.get("baseline_exact_f1", -1.0)
            ),
            "best_update": 0,
            "last_scores": None,
        }
        if state.get("warm_start"):
            if (
                hashes(run / "baseline-model")
                != state["warm_start"]["baseline_model_files"]
            ):
                raise ValueError("Protected baseline checkpoint changed")
            shutil.copytree(run / "baseline-model", run / "checkpoints/model-best")
        else:
            nlp.to_disk(run / "checkpoints/model-best")
    if device == "cuda":
        cuda_model(nlp)
    if (
        nlp.pipe_names != ["transformer", "ner"]
        or sorted(nlp.get_pipe("ner").labels) != state["labels"]
    ):
        raise ValueError("Loaded pipeline/labels differ from the prepared run")
    nlp.meta.update(
        name="sorani_ner_v2",
        version="2.0.0",
        description="Sorani NER V2; original article identities unavailable",
    )
    drive = Path(state["state_drive"])
    wanted = identity(state)
    if (drive / "identity.json").exists() and json.loads(
        (drive / "identity.json").read_text()
    ) != wanted:
        raise ValueError("Drive run identity collision")
    atomic_json(drive / "identity.json", wanted)

    def evaluate(model, dev):
        scores = model.evaluate(
            _examples(model, dev), batch_size=cfg["nlp"]["batch_size"]
        )
        model.meta["performance"] = scores
        return scores

    def checkpoint(model, opt, position, improved):
        if improved:
            # Convenience output is rebuilt from a verified generation on resume.
            target = run / "checkpoints/model-best"
            shutil.rmtree(target)
            model.to_disk(target)
        # Fail before staging another large generation if either filesystem is
        # already too full. Existing complete generations remain untouched.
        model_bytes = sum(
            p.stat().st_size
            for p in (run / "checkpoints/model-best").rglob("*")
            if p.is_file()
        )
        moment_bytes = sum(
            v.nbytes for slot in (opt.mom1, opt.mom2) for v in slot.values()
        )
        required = 2 * model_bytes + moment_bytes + 64 * 1024**2
        for location, needed in ((run, required + model_bytes), (drive, required)):
            available = shutil.disk_usage(location).free
            if available < needed:
                raise OSError(
                    f"Insufficient checkpoint space at {location}: need {needed} bytes, have {available}; previous generations preserved"
                )
        state.update(
            status="trained" if position["finished"] else "running",
            global_updates=position["updates"],
            best_dev_f1=position["best_f1"],
            best_update=position["best_update"],
            last_gpu_memory=memory() if device == "cuda" else {},
        )
        atomic_json(run / "run.json", state)

        def writer(stage):
            model.to_disk(stage / "checkpoints/model-last")
            shutil.copytree(
                run / "checkpoints/model-best", stage / "checkpoints/model-best"
            )
            save_state(
                stage / "training-state", model, opt, position, cuda=device == "cuda"
            )
            for name in (
                "run.json",
                "config.cfg",
                "runtime-policy.json",
                "code-manifest.json",
                "environment.json",
                "dependencies.txt",
                "effective-changes.json",
                "preflight-smoke.json",
                "preflight-stress.json",
                "preflight-benchmark.json",
                "baseline.json",
                "numerics.jsonl",
                "metrics.jsonl",
            ):
                if (run / name).is_file():
                    shutil.copy2(run / name, stage / name)
            with (stage / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "updates": position["updates"],
                            "epoch": position["epoch"],
                            "dev": position["last_scores"],
                            "best_f1": position["best_f1"],
                            "baseline_f1": state.get("baseline_f1"),
                            "lr": opt.learn_rate,
                        }
                    )
                    + "\n"
                )

        local = publish_generation(
            run / "state-checkpoints",
            writer,
            {
                "identity": wanted,
                "labels": state["labels"],
                "updates": position["updates"],
            },
        )
        shutil.copy2(local / "metrics.jsonl", run / "metrics.jsonl")
        replicate_generation(local, drive)
        # Populate conventional model-last for selection/inference only.
        shutil.copytree(
            local / "checkpoints/model-last",
            run / "checkpoints/model-last",
            dirs_exist_ok=True,
        )
        print(
            json.dumps(
                {
                    "updates": position["updates"],
                    "best_f1": position["best_f1"],
                    "generation": local.name,
                }
            ),
            flush=True,
        )

    state["status"] = "running"
    atomic_json(run / "run.json", state)
    try:
        # Generation zero makes interruptions before the first evaluation resumable.
        if not resume:
            checkpoint(nlp, optimizer, progress, False)
        elif progress["finished"]:
            # A terminal local generation may have survived a failed Drive
            # publication. Finalize it without taking another optimizer step.
            checkpoint(nlp, optimizer, progress, False)
            return progress
        fit(
            nlp,
            optimizer,
            records,
            dev_records,
            cfg,
            policy,
            progress,
            evaluate=evaluate,
            checkpoint=checkpoint,
            diagnostic_path=run / "numerics.jsonl",
            device=device,
        )
    except BaseException as error:
        state.update(status="interrupted_or_failed", error_type=type(error).__name__)
        atomic_json(run / "run.json", state)
        raise

"""Real offline tiny-XLM-R warm-start and interrupted-training regressions."""

import copy
import json
import random
from types import SimpleNamespace

import pytest

spacy = pytest.importorskip("spacy")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("spacy_transformers")

from test_ner_colab_numerics import POLICY, tiny_pipeline

from kurdish_nlp.ner.colab_artifacts import atomic_json, hashes
from kurdish_nlp.ner.colab_numerics import configure_model
from kurdish_nlp.ner.continued import batch_plan, fit, load_initial, validate_progress
from kurdish_nlp.ner.training_state import (
    fresh_optimizer,
    latest_generation,
    learning_rate,
    load_state,
    parameter_map,
    publish_generation,
    replicate_generation,
    restore_training,
    save_state,
    validate_generation,
    weight_fingerprints,
)


@pytest.fixture(autouse=True)
def restore_microbatch_patch(monkeypatch):
    from spacy.training import loop

    monkeypatch.setattr(loop, "subdivide_batch", loop.subdivide_batch)


def settings(nlp):
    cfg = nlp.config.copy()
    cfg["training"].update(max_epochs=3, eval_frequency=2, patience=10, dropout=0.15)
    cfg["training"]["batcher"]["size"] = 8
    policy = {
        **POLICY,
        "state_format": 2,
        "checkpoint_frequency": 1,
        "schedule": {"peak": 5e-6, "warmup": 2, "total": 6},
    }
    nlp.config["training"] = cfg["training"]
    return cfg, policy


def records():
    return [
        {
            "id": str(i),
            "tokens": ["a", "b", "a", "b"],
            "tags": ["B-X", "O", "O", "O"],
            "group": str(i),
        }
        for i in range(6)
    ]


def test_schedule_is_positive_conservative_and_has_explicit_boundary():
    schedule = {"peak": 5e-6, "warmup": 200, "total": 2000}
    assert learning_rate(schedule, 0) == 5e-6 / 200
    assert learning_rate(schedule, 199) == 5e-6
    assert learning_rate(schedule, 200) == 5e-6
    assert 0 < learning_rate(schedule, 1999) < 5e-6
    for bad in ({**schedule, "peak": float("nan")}, {**schedule, "warmup": 2000}):
        with pytest.raises(ValueError):
            learning_rate(bad, 0)
    with pytest.raises(ValueError, match="exhausted"):
        learning_rate(schedule, 2000)


@pytest.fixture
def model(tmp_path):
    spacy.require_cpu()
    torch.set_num_threads(1)
    return tiny_pipeline(tmp_path / "encoder")[0]


def test_warm_start_matches_every_source_weight_and_updates_both_components(
    model, tmp_path, monkeypatch
):
    cfg, policy = settings(model)
    # Source represents an already-trained checkpoint, not just architecture.
    from kurdish_nlp.ner.colab import _examples
    from kurdish_nlp.ner.colab_numerics import checked_update

    optimizer = fresh_optimizer(model, policy["schedule"])
    for _ in range(3):
        checked_update(
            model,
            _examples(model, records()[:2]),
            optimizer,
            accumulate=8,
            dropout=0,
            policy=policy,
            expected_device="cpu",
        )
    source = tmp_path / "original"
    model.to_disk(source)
    expected = weight_fingerprints(model)
    import shutil

    run = tmp_path / "new-run"
    shutil.copytree(source, run / "initial-model")
    state = {
        "warm_start": {
            "source_model_files": hashes(source),
            "initial_weights": expected,
        }
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Warm-start reinitialized a trained component")

    monkeypatch.setattr(spacy.language.Language, "initialize", forbidden)
    monkeypatch.setattr(spacy.language.Language, "resume_training", forbidden)
    loaded = load_initial(run, cfg, state)
    configure_model(loaded, policy)
    assert weight_fingerprints(loaded) == expected
    assert loaded.tokenizer.to_bytes() == model.tokenizer.to_bytes()
    assert (
        loaded.get_pipe("ner").moves.to_bytes()
        == model.get_pipe("ner").moves.to_bytes()
    )
    assert loaded.get_pipe("ner").labels == model.get_pipe("ner").labels
    opt = fresh_optimizer(loaded, policy["schedule"])
    assert not opt.mom1 and not opt.mom2 and not opt.nr_update and not opt.schedules
    assert weight_fingerprints(loaded) == expected  # BEFORE first optimizer update.
    checked_update(
        loaded,
        _examples(loaded, records()[:2]),
        opt,
        accumulate=8,
        dropout=0.1,
        policy=policy,
        expected_device="cpu",
    )
    after = weight_fingerprints(loaded)
    assert all(after[k] != expected[k] for k in expected)
    assert hashes(source) == hashes(run / "initial-model")


def interrupted_equivalence(model, tmp_path, *, cuda=False):
    cfg, policy = settings(model)
    rows = records()
    initial = tmp_path / "initial"
    model.to_disk(initial)

    def start():
        m = spacy.load(initial)
        configure_model(m, policy)
        opt = fresh_optimizer(m, policy["schedule"])
        p = {
            "updates": 0,
            "epoch": 0,
            "cursor": 0,
            "words": 0,
            "plan": batch_plan(m, rows, cfg, 0),
            "finished": False,
            "best_f1": 0.8733,
            "best_update": 0,
            "last_scores": None,
        }
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        if cuda:
            torch.cuda.manual_seed_all(789)
        return m, opt, p

    # An evaluator that consumes RNG proves saved state is AFTER evaluation.
    def evaluate(m, dev):
        random.random()
        np.random.random()
        torch.rand(1)
        return {"ents_f": 87.34 if m._ner_v2_update_index >= 4 else 80.0}

    def save(m, opt, p, improved):
        m.to_disk(tmp_path / "saved-model")
        import shutil

        path = tmp_path / "saved-state"
        if path.exists():
            shutil.rmtree(path)
        save_state(path, m, opt, copy.deepcopy(p), cuda=cuda)

    device = "cuda" if cuda else "cpu"
    whole, opt_whole, p_whole = start()
    fit(
        whole,
        opt_whole,
        rows,
        [],
        cfg,
        policy,
        p_whole,
        evaluate=evaluate,
        checkpoint=lambda *a: None,
        device=device,
    )
    whole_weights = weight_fingerprints(whole)
    expected_rng = (random.random(), np.random.random(), torch.rand(1).item())
    part, opt_part, p_part = start()
    fit(
        part,
        opt_part,
        rows,
        [],
        cfg,
        policy,
        p_part,
        evaluate=evaluate,
        checkpoint=save,
        device=device,
        stop_after=3,
    )
    assert p_part["updates"] == 3 and not p_part["finished"]
    old_keys, _ = parameter_map(part)
    # Force fresh IDs and RNG perturbations, like a separate runtime.
    random.random(), np.random.random(), torch.rand(5)
    resumed = spacy.load(tmp_path / "saved-model")
    configure_model(resumed, policy)
    opt_resume = fresh_optimizer(resumed, policy["schedule"])
    p_resume = load_state(tmp_path / "saved-state", resumed, opt_resume)
    new_keys, _ = parameter_map(resumed)
    assert set(old_keys).isdisjoint(set(new_keys))
    assert len(opt_resume.mom1) == len(opt_part.mom1) > 0
    fit(
        resumed,
        opt_resume,
        rows,
        [],
        cfg,
        policy,
        p_resume,
        evaluate=evaluate,
        checkpoint=lambda *a: None,
        device=device,
    )
    assert p_resume == p_whole
    if cuda:
        # CUDA kernels can differ in reduction order: bit equality isn't promised.
        for a, b in zip(
            whole.get_pipe("transformer").model.transformer.parameters(),
            resumed.get_pipe("transformer").model.transformer.parameters(),
        ):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        from thinc.util import to_numpy

        for a, b in zip(
            whole.get_pipe("ner").model.walk(), resumed.get_pipe("ner").model.walk()
        ):
            assert a.param_names == b.param_names
            for name in a.param_names:
                np.testing.assert_allclose(
                    to_numpy(a.get_param(name)),
                    to_numpy(b.get_param(name)),
                    rtol=1e-5,
                    atol=1e-6,
                )
    else:
        assert weight_fingerprints(resumed) == whole_weights
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected_rng
    for slot in ("mom1", "mom2", "nr_update", "last_seen"):

        def normalized(m, opt, slot=slot):
            keys, _ = parameter_map(m)
            return {keys[k]: v for k, v in getattr(opt, slot).items()}

        expected, actual = normalized(whole, opt_whole), normalized(resumed, opt_resume)
        assert expected.keys() == actual.keys()
        for key in expected:
            a, b = expected[key], actual[key]
            if slot in {"mom1", "mom2"}:
                from thinc.util import to_numpy

                if cuda:
                    np.testing.assert_allclose(
                        to_numpy(a), to_numpy(b), rtol=1e-4, atol=1e-7
                    )
                else:
                    np.testing.assert_array_equal(to_numpy(a), to_numpy(b))
            else:
                assert a == b
    assert opt_resume.learn_rate == opt_whole.learn_rate
    scale = lambda m: m.get_pipe("transformer").model.layers[0].shims[0]._grad_scaler
    assert scale(resumed)._scale.item() == scale(whole)._scale.item()
    assert scale(resumed)._growth_tracker.item() == scale(whole)._growth_tracker.item()


def test_interrupted_vs_uninterrupted_exact_cpu(model, tmp_path):
    interrupted_equivalence(model, tmp_path)


def test_cuda_interrupted_vs_uninterrupted(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Needs Colab T4: real FP16/scaler/checkpointed CUDA resume")
    pytest.importorskip("cupy")
    from thinc.api import set_gpu_allocator

    from kurdish_nlp.ner.colab_dlpack import install_cupy_dlpack_compatibility

    spacy.require_gpu(0)
    set_gpu_allocator("pytorch")
    install_cupy_dlpack_compatibility()
    try:
        torch.set_num_threads(1)
        nlp, _ = tiny_pipeline(tmp_path / "cuda-encoder", cuda=True)
        interrupted_equivalence(nlp, tmp_path, cuda=True)
    finally:
        spacy.require_cpu()
        torch.cuda.empty_cache()


def test_integrity_publication_and_restore(model, tmp_path, monkeypatch):
    _cfg, policy = settings(model)
    opt = fresh_optimizer(model, policy["schedule"])
    position = {"updates": 0}
    root = tmp_path / "generations"

    def writer(stage):
        model.to_disk(stage / "checkpoints/model-last")
        model.to_disk(stage / "checkpoints/model-best")
        save_state(stage / "training-state", model, opt, position, cuda=False)

    meta = {"identity": {"run_id": "fixture"}, "updates": 0, "labels": ["X"]}
    good = publish_generation(root, writer, meta)
    first_pointer = (root / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="compatibility"):
        validate_generation(good, {"run_id": "different"})

    def fail(stage):
        (stage / "unfinished").write_text("partial")
        raise OSError("disconnect")

    with pytest.raises(OSError, match="disconnect"):
        publish_generation(root, fail, meta)
    assert (root / "latest.json").read_bytes() == first_pointer
    validate_generation(good)
    second = replicate_generation(good, root)
    assert good.exists() and second.exists()
    restored = restore_training(root, tmp_path / "restored")
    validate_generation(latest_generation(restored / "state-checkpoints"))
    with pytest.raises(FileExistsError):
        restore_training(root, restored)
    (second / "training-state/state.json").write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        restore_training(root, tmp_path / "blocked")
    validate_generation(good)  # Verified previous generation still usable.


def test_weights_only_never_passes_true_resume_validation(tmp_path):
    atomic_json(
        tmp_path / "COMPLETE.json",
        {"format": 1, "resume_state": "weights_only_no_optimizer_or_scheduler"},
    )
    with pytest.raises(ValueError, match="weights-only"):
        validate_generation(tmp_path)


def test_sampler_rejects_duplicates_and_invalid_cursor(model):
    cfg, _ = settings(model)
    plan = batch_plan(model, records(), cfg, 0)
    assert plan == batch_plan(model, records(), cfg, 0)
    progress = {
        "updates": 1,
        "epoch": 0,
        "cursor": 99,
        "plan": plan,
        "best_update": 0,
        "best_f1": 0.8733,
    }
    with pytest.raises(ValueError, match="sampler"):
        validate_progress(progress, records(), {"total": 6})


def test_checkpoint_directory_created_only_after_anti_resume_gate(
    tmp_path, monkeypatch
):
    from kurdish_nlp.ner import colab, continued

    args = SimpleNamespace(run=tmp_path)
    state = {"status": "prepared", "exact_resume_supported": True}
    monkeypatch.setattr(colab, "gate", lambda a: state)
    monkeypatch.setattr(colab, "load_run", lambda a: (state, {}, {}))
    monkeypatch.setattr(colab, "_window_preflight", lambda *a: None)

    def train(*a):
        assert (tmp_path / "checkpoints").is_dir()

    monkeypatch.setattr(continued, "train_run", train)
    colab.full_train(args)
    with pytest.raises(FileExistsError, match="pseudo-resume"):
        colab.full_train(args)


def test_production_runner_interrupt_restore_resume_preserves_baseline(
    model, tmp_path, monkeypatch
):
    """Exercise the actual orchestrator, disk generations and dev evaluator on CPU."""
    import hashlib
    import shutil

    from kurdish_nlp.ner import colab, continued
    from kurdish_nlp.ner.build import write_jsonl
    from kurdish_nlp.ner.dataset import sha256
    from kurdish_nlp.ner.training import convert_records
    from kurdish_nlp.ner.training_state import latest_generation

    run, data, drive = tmp_path / "run", tmp_path / "data", tmp_path / "drive"
    run.mkdir()
    data.mkdir()
    (run / "checkpoints").mkdir()
    cfg, policy = settings(model)
    policy["schedule"]["total"] = 6
    cfg.to_disk(run / "config.cfg")
    atomic_json(run / "runtime-policy.json", policy)
    model.to_disk(run / "initial-model")
    shutil.copytree(run / "initial-model", run / "baseline-model")
    baseline = hashes(run / "baseline-model")
    for split in ("train", "dev"):
        write_jsonl(data / f"{split}.jsonl", records())
    code = {"format": 1, "files": {}, "base_git_commit": "fixture"}
    code["content_sha256"] = hashlib.sha256(
        json.dumps(code, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(run / "code-manifest.json", code)
    lock = tmp_path / "lock.json"
    lock.write_text("synthetic frozen lock")
    for split in ("train", "dev"):
        convert_records(records(), run / f"{split}.spacy", spacy.blank("xx"))
    state = {
        "run_id": "sorani-ner-v2-fixture",
        "dataset_lock_sha256": sha256(lock),
        "config_sha256": sha256(run / "config.cfg"),
        "runtime_policy_sha256": sha256(run / "runtime-policy.json"),
        "corpora_sha256": {s: sha256(run / f"{s}.spacy") for s in ("train", "dev")},
        "working_tree_content_sha256": code["content_sha256"],
        "exact_resume_supported": True,
        "state_drive": str(drive),
        "labels": ["X"],
        "status": "prepared",
        "baseline_f1": 1.0,  # Synthetic perfect baseline cannot be beaten.
        "warm_start": {
            "source_model_files": hashes(run / "initial-model"),
            "initial_weights": weight_fingerprints(model),
            "baseline_model_files": baseline,
        },
    }
    atomic_json(run / "run.json", state)
    args = SimpleNamespace(run=run, data=data)
    real_update = continued.checked_update
    calls = []

    def interrupt(*a, **kw):
        calls.append(1)
        if len(calls) == 4:
            raise OSError("simulated interruption inside next update")
        return real_update(*a, **kw)

    monkeypatch.setattr(continued, "checked_update", interrupt)
    with pytest.raises(OSError, match="interruption"):
        continued.train_run(args, state, cfg, device="cpu")
    assert (
        json.loads((run / "run.json").read_text())["status"] == "interrupted_or_failed"
    )
    marker = validate_generation(latest_generation(drive))
    assert marker["updates"] == 3
    restored = restore_training(drive, tmp_path / "restored")
    sealed = latest_generation(restored / "state-checkpoints")
    sealed_files = hashes(sealed)
    monkeypatch.setattr(colab, "gpu_environment", dict)
    monkeypatch.setattr(colab, "verify", lambda *a: {"labels": ["X"]})
    recovery_args = SimpleNamespace(
        run=restored, data=data, lock=lock, source=None, command="resume"
    )
    restored_state, restored_cfg, _ = colab.load_run(recovery_args)
    assert all((restored / f"{s}.spacy").exists() for s in ("train", "dev"))
    assert hashes(sealed) == sealed_files
    monkeypatch.setattr(continued, "checked_update", real_update)
    continued.train_run(
        SimpleNamespace(run=restored, data=data),
        restored_state,
        restored_cfg,
        resume=True,
        device="cpu",
    )
    final = json.loads((restored / "run.json").read_text())
    assert final["status"] == "trained" and final["global_updates"] == 6
    assert hashes(restored / "checkpoints/model-best") == baseline
    assert final["best_update"] == 0 and final["best_dev_f1"] == 1.0
    assert validate_generation(latest_generation(drive))["updates"] == 6
    with pytest.raises(ValueError, match="terminal"):
        continued.train_run(
            SimpleNamespace(run=restored, data=data),
            final,
            cfg,
            resume=True,
            device="cpu",
        )


def test_partial_accumulation_cannot_be_saved(model, tmp_path):
    _cfg, policy = settings(model)
    opt = fresh_optimizer(model, policy["schedule"])
    model._ner_v2_in_update = True
    with pytest.raises(ValueError, match="incomplete effective-batch"):
        save_state(tmp_path / "blocked", model, opt, {"updates": 0}, cuda=False)
    assert not (tmp_path / "blocked").exists()
    model._ner_v2_in_update = False
    node = next(n for n in model.get_pipe("ner").model.walk() if n.param_names)
    name = node.param_names[0]
    node.set_grad(name, node.ops.xp.ones_like(node.get_param(name)))
    with pytest.raises(ValueError, match="accumulated gradients"):
        save_state(tmp_path / "blocked", model, opt, {"updates": 0}, cuda=False)


def test_prepare_continued_provenance_config_and_immutable_source(
    model, tmp_path, monkeypatch
):
    import hashlib
    import shutil

    from spacy.util import load_config

    from kurdish_nlp.ner import colab
    from kurdish_nlp.ner.build import write_jsonl
    from kurdish_nlp.ner.dataset import sha256

    repo = tmp_path / "repo"
    config_dir = repo / "configs/ner"
    config_dir.mkdir(parents=True)
    from pathlib import Path

    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs/ner/xlmr-v2.cfg",
        config_dir / "xlmr-v2.cfg",
    )
    monkeypatch.chdir(repo)
    lock = config_dir / "lock.json"
    lock.write_text("synthetic frozen release")
    data = repo / "data"
    data.mkdir()
    for split in ("train", "dev"):
        write_jsonl(data / f"{split}.jsonl", records())
    source = tmp_path / "step-00006000-578fdaf6"
    (source / "checkpoints").mkdir(parents=True)
    model.meta["performance"] = {"ents_f": 87.334}
    for role in ("model-best", "model-last"):
        model.to_disk(source / "checkpoints" / role)
    atomic_json(
        source / "run.json", {"dataset_lock_sha256": sha256(lock), "run_id": "original"}
    )
    atomic_json(source / "runtime-policy.json", POLICY)
    atomic_json(
        source / "COMPLETE.json",
        {
            "format": 1,
            "run_id": "original",
            "step": 6000,
            "labels": ["X"],
            "resume_state": "weights_only_no_optimizer_or_scheduler",
            "files": hashes(source),
        },
    )
    code = {"format": 1, "base_git_commit": "fixture", "files": {}}
    code["content_sha256"] = hashlib.sha256(
        json.dumps(code, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(repo / "code-manifest.json", code)
    before = hashes(source)
    monkeypatch.setattr(
        colab.subprocess,
        "check_output",
        lambda *a, **kw: "synthetic dependency freeze\n",
    )
    monkeypatch.setattr(colab, "gpu_environment", dict)
    monkeypatch.setattr(colab, "LOCK_SHA256", sha256(lock))
    monkeypatch.setattr(colab, "verify", lambda *a: {"labels": ["X"]})
    monkeypatch.setattr(colab, "validate_encoder", lambda *a: {"revision": "synthetic"})
    args = SimpleNamespace(
        run=tmp_path / "sorani-ner-v2-continued-fixture",
        drive=tmp_path / "drive/sorani-ner-v2-continued-fixture",
        snapshot=source,
        role="model-last",
        original_run="original",
        generation=source.name,
        encoder=tmp_path / "encoder",
        data=data,
        lock=lock,
        source=None,
        mixed_precision=False,
        gradient_checkpointing=True,
        accumulate=8,
        eval_batch=8,
        updates=6,
        warmup=2,
        learning_rate=5e-6,
        patience=4,
        eval_frequency=2,
        checkpoint_frequency=2,
        baseline_f1=0.8733,
    )
    colab.prepare(args)
    state = json.loads((args.run / "run.json").read_text())
    assert state["warm_start"]["original_run"] == "original"
    assert state["warm_start"]["source_complete_sha256"] == sha256(
        source / "COMPLETE.json"
    )
    assert state["warm_start"]["initial_weights"] == weight_fingerprints(model)
    assert state["warm_start"]["optimizer_restored"] is False
    assert state["baseline_f1"] == 0.8733 and state["exact_resume_supported"] is True
    assert state["baseline_exact_f1"] == pytest.approx(0.87334)
    assert not (args.run / "checkpoints").exists()
    cfg = load_config(args.run / "config.cfg")
    assert cfg["training"]["optimizer"]["learn_rate"] == pytest.approx(2.5e-6)
    changes = json.loads((args.run / "effective-changes.json").read_text())
    assert changes["training.patience"]["effective"] == 4
    assert hashes(source) == before
    with pytest.raises(FileExistsError, match="immutable"):
        colab.prepare(args)

    # Stock notebook preparation remains weights-only unless explicitly opted in.
    args.snapshot = None
    args.run = tmp_path / "sorani-ner-v2-fresh-legacy"
    args.drive = tmp_path / "drive/sorani-ner-v2-fresh-legacy"
    colab.prepare(args)
    legacy_cfg = load_config(args.run / "config.cfg")
    assert (
        legacy_cfg["training"]["optimizer"]["learn_rate"]["@schedules"]
        == "warmup_linear.v1"
    )
    assert legacy_cfg["training"]["max_steps"] == 20000
    assert (
        json.loads((args.run / "run.json").read_text())["exact_resume_supported"]
        is False
    )
    assert "state_format" not in json.loads(
        (args.run / "runtime-policy.json").read_text()
    )
    args.run = tmp_path / "sorani-ner-v2-fresh-resumable"
    args.drive = tmp_path / "drive/sorani-ner-v2-fresh-resumable"
    args.resumable = True
    colab.prepare(args)
    assert (
        json.loads((args.run / "run.json").read_text())["exact_resume_supported"]
        is True
    )
    assert (
        json.loads((args.run / "runtime-policy.json").read_text())["state_format"] == 2
    )

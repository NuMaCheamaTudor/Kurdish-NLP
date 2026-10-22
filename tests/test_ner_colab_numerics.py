"""Offline numerical regressions plus real CUDA AMP tests (no model downloads).

CPU tests exercise the pinned spaCy/HF/Thinc backward and optimizer. Only the
scaler's CUDA-only multiply is emulated in CPU overflow tests; CUDA tests use
unmodified autocast and scaling and skip honestly when hardware is unavailable.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spacy = pytest.importorskip("spacy")
pytest.importorskip("spacy_transformers")

from kurdish_nlp.ner.colab import install_microbatch_guard
from kurdish_nlp.ner.colab_numerics import (
    DeferredParameterUnscale,
    NumericalFailure,
    checked_update,
    configure_model,
    gradient_report,
    install_training_loop,
)

REPO = Path(__file__).resolve().parents[1]
POLICY = {"version": 1, "gradient_checkpointing": True, "max_overflow_retries": 2}


def tiny_pipeline(path, *, cuda=False):
    from spacy.training import Example
    from spacy.util import load_config, load_model_from_config
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    path.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=512,
    ).save_pretrained(path)
    XLMRobertaModel(
        XLMRobertaConfig(
            vocab_size=6,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=514,
        )
    ).save_pretrained(path)
    cfg = load_config(REPO / "configs/ner/xlmr-v2.cfg")
    cfg["components"]["transformer"]["model"]["name"] = str(path)
    cfg["components"]["transformer"]["model"]["mixed_precision"] = cuda
    cfg["components"]["transformer"]["model"]["grad_scaler_config"] = {"init_scale": 128.0}
    nlp = load_model_from_config(cfg, auto_fill=True)
    batch = [
        Example.from_dict(nlp.make_doc(text), {"entities": [(0, 1, "X")]})
        for text in ("a b " * 100, "b a b")
    ]
    nlp.initialize(lambda: batch)
    install_microbatch_guard()
    configure_model(nlp, POLICY)
    return nlp, batch


class RecordingOptimizer:
    def __init__(self, optimizer, scaler):
        self.original, self.scaler = optimizer, scaler
        self.calls, self.schedules = [], 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, key, *args, **kwargs):
        self.calls.append((key, self.scaler._scale.item()))
        return self.original(key, *args, **kwargs)

    def step_schedules(self):
        self.schedules += 1
        self.original.step_schedules()


@pytest.fixture
def pipeline(tmp_path):
    spacy.require_cpu()
    torch.set_num_threads(1)
    return tiny_pipeline(tmp_path / "encoder")


def do_update(nlp, batch, optimizer, tmp_path, **kwargs):
    return checked_update(
        nlp,
        batch,
        optimizer,
        accumulate=8,
        dropout=0.0,
        policy=POLICY,
        diagnostic_path=tmp_path / "numerics.jsonl",
        expected_device="cpu",
        **kwargs,
    )


def test_unscale_once_matches_sum_and_reproduces_upstream_accumulation_bug():
    from thinc.api import PyTorchGradScaler

    model = torch.nn.Linear(1, 1, bias=False)
    scaler = PyTorchGradScaler(enabled=True, init_scale=128.0)
    model.weight.grad = torch.tensor([[2.0 * 128]])
    scaler.unscale([model.weight.grad])
    model.weight.grad.add_(3.0 * 128)
    scaler.unscale([model.weight.grad])
    assert model.weight.grad.item() == 2.0 / 128 + 3.0
    model.weight.grad = torch.tensor([[2.0 * 128]])
    proxy = DeferredParameterUnscale(scaler, model)
    proxy.unscale([model.weight.grad])
    model.weight.grad.add_(3.0 * 128)
    proxy.unscale([model.weight.grad])
    assert model.weight.grad.item() == 5.0 * 128
    proxy.finish()
    assert model.weight.grad.item() == 5.0
    with pytest.raises(RuntimeError, match="before effective-batch"):
        proxy.update()


def test_named_report_separates_missing_device_nan_inf_and_unused_pooler():
    model = torch.nn.Module()
    model.register_parameter("missing", torch.nn.Parameter(torch.ones(1)))
    model.register_parameter("bad", torch.nn.Parameter(torch.ones(3)))
    model.bad.grad = torch.tensor([float("nan"), float("inf"), -float("inf")])
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    model.pooler = torch.nn.Module()
    model.pooler.dense = torch.nn.Linear(1, 1)
    report = gradient_report(model)
    assert report["missing"] == ["missing"]
    assert report["frozen"] == ["frozen"]
    assert report["expected_unused"] == ["pooler.dense.weight", "pooler.dense.bias"]
    assert report["wrong_device"][0]["name"] == "bad"
    assert report["nonfinite"][0] == {
        "name": "bad",
        "shape": [3],
        "dtype": "torch.float32",
        "device": "cpu",
        "nan": 1,
        "positive_inf": 1,
        "negative_inf": 1,
    }


def test_actual_pipeline_long_windows_checkpointing_and_reload(pipeline, tmp_path):
    nlp, batch = pipeline
    hf = nlp.get_pipe("transformer").model.transformer
    before = hf.encoder.layer[0].attention.self.query.weight.detach().clone()
    optimizer = nlp.create_optimizer()
    for _ in range(3):
        result = do_update(nlp, batch, optimizer, tmp_path)
    assert result["records"] == 2 and result["tokens"] == 203
    assert result["attempts"][0]["microbatch_records"] == [1, 1]
    assert result["attempts"][0]["deferred_backward_calls"] == 2
    assert result["attempts"][0]["gradients"]["missing"] == []
    assert not torch.equal(before, hf.encoder.layer[0].attention.self.query.weight)
    assert hf.is_gradient_checkpointing
    doc = nlp(batch[0].predicted.text)
    assert len(doc) == 200
    assert len(doc._.trf_data.wordpieces.lengths) > 1
    nlp.to_disk(tmp_path / "saved")
    loaded = spacy.load(tmp_path / "saved")
    configure_model(loaded, POLICY)
    assert loaded.get_pipe("transformer").model.transformer.is_gradient_checkpointing
    assert len(loaded(batch[0].predicted.text)) == 200


def emulate_cpu_scaling(nlp, monkeypatch):
    scaler = nlp.get_pipe("transformer").model.layers[0].shims[0]._grad_scaler
    scaler._enabled = True

    def scale(tensors, inplace=False):
        return [t.mul_(scaler._scale) if inplace else t * scaler._scale for t in tensors]

    monkeypatch.setattr(scaler, "scale", scale)
    return scaler


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")])
def test_amp_backoff_replays_complete_batch_before_both_optimizers(
    pipeline, tmp_path, monkeypatch, bad_value
):
    nlp, batch = pipeline
    scaler = emulate_cpu_scaling(nlp, monkeypatch)
    weight = nlp.get_pipe("transformer").model.transformer.embeddings.word_embeddings.weight
    # Deliberately create a backward-only nonfinite while forward/loss/head stay finite.
    hook = weight.register_hook(
        lambda g: torch.full_like(g, bad_value) if scaler._scale.item() > 64 else g
    )
    optimizer = RecordingOptimizer(nlp.create_optimizer(), scaler)
    result = do_update(nlp, batch, optimizer, tmp_path)
    hook.remove()
    assert result["overflow_retries"] == 1
    assert result["optimizer_updates"] == 1
    assert optimizer.calls and all(scale == 64 for _, scale in optimizer.calls)
    assert optimizer.schedules == 1
    shim = nlp.get_pipe("transformer").model.layers[0].shims[0]
    assert any(key[0] == shim.id for key, _ in optimizer.calls)
    assert any(key[0] != shim.id for key, _ in optimizer.calls)
    assert [a["outcome"] for a in result["attempts"]] == [
        "overflow_replayed",
        "applied_finite_update",
    ]
    assert all(a["microbatch_records"] == [1, 1] for a in result["attempts"])
    assert result["attempts"][0]["replay_seed"] == result["attempts"][1]["replay_seed"]
    assert (
        result["attempts"][0]["gradients"]["nonfinite"][0]["name"]
        == "embeddings.word_embeddings.weight"
    )


def test_persistent_nonfinite_fails_without_any_optimizer_or_scheduler_step(
    pipeline, tmp_path, monkeypatch
):
    nlp, batch = pipeline
    emulate_cpu_scaling(nlp, monkeypatch)
    hf = nlp.get_pipe("transformer").model.transformer
    hook = hf.embeddings.word_embeddings.weight.register_hook(
        lambda g: torch.full_like(g, float("inf"))
    )
    opt = RecordingOptimizer(
        nlp.create_optimizer(),
        hf and nlp.get_pipe("transformer").model.layers[0].shims[0]._grad_scaler,
    )
    with pytest.raises(NumericalFailure) as failure:
        do_update(nlp, batch, opt, tmp_path)
    hook.remove()
    assert failure.value.diagnostics["category"] == "persistent_amp_nonfinite"
    assert len(failure.value.diagnostics["attempts"]) == 3
    assert opt.calls == [] and opt.schedules == 0
    events = [json.loads(line) for line in (tmp_path / "numerics.jsonl").read_text().splitlines()]
    assert len(events) == 3


def test_unexpected_missing_gradient_is_fatal(pipeline, tmp_path):
    nlp, batch = pipeline
    hf = nlp.get_pipe("transformer").model.transformer
    hf.register_parameter("orphan", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(NumericalFailure) as failure:
        do_update(nlp, batch, nlp.create_optimizer(), tmp_path)
    assert failure.value.diagnostics["category"] == "missing_gradient"
    assert "orphan" in failure.value.diagnostics["attempts"][-1]["gradients"]["missing"]


def test_nonfinite_forward_is_fatal_and_logged(pipeline, tmp_path):
    nlp, batch = pipeline
    hf = nlp.get_pipe("transformer").model.transformer

    def corrupt(module, inputs, output):
        output.last_hidden_state.fill_(float("nan"))

    hook = hf.register_forward_hook(corrupt)
    with pytest.raises(NumericalFailure, match="nonfinite_forward"):
        do_update(nlp, batch, nlp.create_optimizer(), tmp_path)
    hook.remove()
    assert (
        json.loads((tmp_path / "numerics.jsonl").read_text())["failure"]["category"]
        == "nonfinite_forward"
    )


def test_full_training_uses_checked_updates_and_preserves_selection(
    pipeline, tmp_path, monkeypatch
):
    from spacy.training import loop

    nlp, batch = pipeline
    original = loop.train_while_improving
    monkeypatch.setattr(loop, "train_while_improving", original)
    # This CPU test replaces only device selection at the actual checked step.
    import kurdish_nlp.ner.colab_numerics as numerics

    checked = numerics.checked_update
    monkeypatch.setattr(
        numerics,
        "checked_update",
        lambda *a, **kw: checked(*a, **kw, expected_device="cpu"),
    )
    install_training_loop(POLICY, tmp_path / "numerics.jsonl")
    results = list(
        loop.train_while_improving(
            nlp,
            nlp.create_optimizer(),
            [(0, batch)] * 4,
            lambda: (1.0, {"ents_f": 1.0}),
            dropout=0.0,
            eval_frequency=1,
            accumulate_gradient=8,
            patience=2,
            max_steps=100,
            exclude=[],
            annotating_components=[],
            before_update=None,
        )
    )
    assert [info["step"] for _, info, _ in results] == [0, 1, 2]
    assert all(is_best for _, _, is_best in results)
    assert len((tmp_path / "numerics.jsonl").read_text().splitlines()) == 3


@pytest.fixture
def cuda_pipeline(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: actual FP16 autocast/scaler execution requires Colab")
    pytest.importorskip("cupy")
    from thinc.api import set_gpu_allocator

    spacy.require_gpu(0)
    set_gpu_allocator("pytorch")
    from kurdish_nlp.ner.colab_dlpack import install_cupy_dlpack_compatibility

    install_cupy_dlpack_compatibility()
    try:
        yield tiny_pipeline(tmp_path / "cuda-encoder", cuda=True)
    finally:
        spacy.require_cpu()
        torch.cuda.empty_cache()


def test_cuda_amp_checkpointed_long_windows_update_and_reload(cuda_pipeline, tmp_path):
    nlp, batch = cuda_pipeline
    hf = nlp.get_pipe("transformer").model.transformer
    assert all(p.is_cuda for p in hf.parameters())
    result = checked_update(
        nlp,
        batch,
        nlp.create_optimizer(),
        accumulate=8,
        dropout=0.0,
        policy=POLICY,
        diagnostic_path=tmp_path / "numerics.jsonl",
    )
    assert result["optimizer_updates"] == 1
    assert not result["attempts"][-1]["gradients"]["nonfinite"]
    assert result["attempts"][-1]["scaler"]["enabled"]
    nlp.to_disk(tmp_path / "saved")
    loaded = spacy.load(tmp_path / "saved")
    configure_model(loaded, POLICY)
    assert loaded.get_pipe("transformer").model.transformer.is_gradient_checkpointing
    checked_update(
        loaded,
        batch,
        loaded.create_optimizer(),
        accumulate=8,
        dropout=0.0,
        policy=POLICY,
    )


def test_cuda_real_scaler_overflow_replay(cuda_pipeline, tmp_path):
    nlp, batch = cuda_pipeline
    model = nlp.get_pipe("transformer").model
    scaler = model.layers[0].shims[0]._grad_scaler
    hook = model.transformer.embeddings.word_embeddings.weight.register_hook(
        lambda g: torch.full_like(g, float("inf")) if scaler._scale.item() > 64 else g
    )
    result = checked_update(
        nlp,
        batch,
        nlp.create_optimizer(),
        accumulate=8,
        dropout=0.0,
        policy=POLICY,
        diagnostic_path=tmp_path / "numerics.jsonl",
    )
    hook.remove()
    assert result["overflow_retries"] >= 1
    assert result["optimizer_updates"] == 1
    assert result["attempts"][-1]["outcome"] == "applied_finite_update"


def test_fp32_nonfinite_is_fatal_without_amp_backoff(pipeline, tmp_path):
    nlp, batch = pipeline
    hf = nlp.get_pipe("transformer").model.transformer
    hook = hf.embeddings.word_embeddings.weight.register_hook(
        lambda g: torch.full_like(g, float("nan"))
    )
    scaler = nlp.get_pipe("transformer").model.layers[0].shims[0]._grad_scaler
    optimizer = RecordingOptimizer(nlp.create_optimizer(), scaler)
    with pytest.raises(NumericalFailure) as failure:
        do_update(nlp, batch, optimizer, tmp_path)
    hook.remove()
    assert failure.value.diagnostics["category"] == "unhandled_nonfinite_gradient"
    assert optimizer.calls == [] and optimizer.schedules == 0


def test_nonfinite_unscaled_ner_gradient_is_fatal_without_amp_replay(
    pipeline, tmp_path, monkeypatch
):
    nlp, batch = pipeline
    scaler = emulate_cpu_scaling(nlp, monkeypatch)
    update = nlp.update

    def corrupt(*args, **kwargs):
        result = update(*args, **kwargs)
        node = next(
            node
            for node in nlp.get_pipe("ner").model.walk()
            if any(node.has_grad(name) for name in node.param_names)
        )
        name = next(name for name in node.param_names if node.has_grad(name))
        node.get_grad(name).fill(float("nan"))
        return result

    monkeypatch.setattr(nlp, "update", corrupt)
    optimizer = RecordingOptimizer(nlp.create_optimizer(), scaler)
    with pytest.raises(NumericalFailure) as failure:
        do_update(nlp, batch, optimizer, tmp_path)
    assert failure.value.diagnostics["category"] == "nonfinite_loss_or_unscaled_ner_gradient"
    assert optimizer.calls == [] and optimizer.schedules == 0
    assert scaler._scale.item() == 128


def test_window_guard_rejects_truncation_and_preserves_exact_objects():
    from importlib import import_module
    from types import SimpleNamespace

    from kurdish_nlp.ner.colab_numerics import install_window_guard

    install_window_guard()
    module = import_module("spacy_transformers.layers.transformer_model")
    wordpieces = SimpleNamespace(lengths=[512], input_ids=torch.zeros((1, 512)))
    align = object()
    result = module.truncate_oversize_splits(wordpieces, align, 512)
    assert result[0] is wordpieces and result[1] is align
    wordpieces.lengths = [513]
    wordpieces.input_ids = torch.zeros((1, 513))
    with pytest.raises(NumericalFailure, match="window_would_truncate"):
        module.truncate_oversize_splits(wordpieces, align, 512)

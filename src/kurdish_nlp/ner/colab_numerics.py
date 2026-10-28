"""Checked, atomic effective-batch updates for the pinned spaCy/Thinc stack.

Thinc 8.3.11 unscales inside each backward callback. Its parameter unscale is
deferred here until ALL microbatches have accumulated, while input gradients
retain Thinc's immediate-unscale behavior. No dependency files are modified.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path


class NumericalFailure(RuntimeError):
    def __init__(self, category, diagnostics):
        self.diagnostics = {"category": category, **diagnostics}
        super().__init__(f"{category}: inspect named gradient/scaler diagnostics in numerics.jsonl")


def verify_stack():
    from importlib.metadata import version

    for name, expected in (
        ("spacy", "3.8.7"),
        ("thinc", "8.3.11"),
        ("spacy-transformers", "1.3.9"),
        ("transformers", "4.49.0"),
    ):
        if version(name) != expected:
            raise RuntimeError(f"Numerical adapter requires {name}=={expected}")


def tensor_finite(value):
    import torch

    return all(
        bool(torch.isfinite(chunk).all()) for chunk in value.detach().reshape(-1).split(1048576)
    )


def scaler_state(scaler):
    return {
        "enabled": bool(scaler._enabled),
        "scale": float(scaler._scale.item()),
        "found_inf": bool(scaler.found_inf),
    }


def gradient_report(model, *, expected_device="cuda", phase="unscaled"):
    """Report every missing/wrong-device/non-finite parameter without hiding None.

    AutoModel's unused XLM-R pooler is not connected to last_hidden_state, the
    only output spaCy backpropagates. Only those two exact names are exempt.
    """
    import torch

    report = {
        "phase": phase,
        "present": 0,
        "finite": 0,
        "missing": [],
        "expected_unused": [],
        "frozen": [],
        "wrong_device": [],
        "nonfinite": [],
        "nonzero_tensors": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            report["frozen"].append(name)
            continue
        grad = parameter.grad
        if grad is None:
            key = (
                "expected_unused"
                if name in {"pooler.dense.weight", "pooler.dense.bias"}
                else "missing"
            )
            report[key].append(name)
            continue
        report["present"] += 1
        metadata = {
            "name": name,
            "shape": list(grad.shape),
            "dtype": str(grad.dtype),
            "device": str(grad.device),
        }
        if grad.device.type != expected_device:
            report["wrong_device"].append(metadata)
        if tensor_finite(grad):
            report["finite"] += 1
            report["nonzero_tensors"] += int(
                any(bool(torch.count_nonzero(chunk)) for chunk in grad.reshape(-1).split(1048576))
            )
        else:
            # Do not materialize an enormous array of bad indices for embeddings.
            report["nonfinite"].append(
                {
                    **metadata,
                    "nan": sum(
                        int(torch.isnan(c).sum().item()) for c in grad.reshape(-1).split(1048576)
                    ),
                    "positive_inf": sum(
                        int(torch.isposinf(c).sum().item()) for c in grad.reshape(-1).split(1048576)
                    ),
                    "negative_inf": sum(
                        int(torch.isneginf(c).sum().item()) for c in grad.reshape(-1).split(1048576)
                    ),
                }
            )
    return report


class DeferredParameterUnscale:
    """Proxy only during an effective batch, restored before finish_update()."""

    def __init__(self, original, model):
        self.original, self.model = original, model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def unscale(self, tensors):
        self.calls += 1
        param_grads = {id(p.grad) for p in self.model.parameters() if p.grad is not None}
        inputs = [g for g in tensors if id(g) not in param_grads]
        if inputs:
            self.original.unscale(inputs)
        return self.original.found_inf

    def finish(self):
        self.original.unscale([p.grad for p in self.model.parameters() if p.grad is not None])

    def update(self):
        raise RuntimeError("Optimizer/scaler update attempted before effective-batch validation")


def thinc_report(nlp, *, parameters=False):
    failures = []
    seen = set()
    for component_name, component in nlp.pipeline:
        for node in component.model.walk():
            if node.id in seen:
                continue
            seen.add(node.id)
            for name in node.param_names:
                if not parameters and not node.has_grad(name):
                    continue
                value = node.get_param(name) if parameters else node.get_grad(name)
                if value is not None and not bool(node.ops.xp.isfinite(value).all()):
                    failures.append(
                        {
                            "component": component_name,
                            "node": node.name,
                            "node_id": node.id,
                            "parameter": name,
                            "kind": "parameter" if parameters else "gradient",
                        }
                    )
    return failures


def clear_gradients(nlp, hf):
    hf.zero_grad(set_to_none=True)
    seen = set()
    for _, component in nlp.pipeline:
        for node in component.model.walk():
            if node.id in seen:
                continue
            seen.add(node.id)
            for name in node.param_names:
                if node.has_grad(name):
                    node.get_grad(name).fill(0)


def emit(path, event):
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()


def install_window_guard():
    """Fail before spaCy's built-in oversize helper could discard wordpieces."""
    from importlib import import_module

    verify_stack()
    module = import_module("spacy_transformers.layers.transformer_model")
    if getattr(module.truncate_oversize_splits, "_ner_v2_no_truncation", False):
        return

    def preserve(wordpieces, align, max_length):
        maximum = max(wordpieces.lengths, default=0)
        if maximum > max_length or wordpieces.input_ids.shape[1] > max_length:
            raise NumericalFailure(
                "window_would_truncate",
                {
                    "max_wordpieces": maximum,
                    "capacity": max_length,
                    "padded_width": int(wordpieces.input_ids.shape[1]),
                },
            )
        return wordpieces, align

    preserve._ner_v2_no_truncation = True
    module.truncate_oversize_splits = preserve


def configure_model(nlp, policy, diagnostic_path=None):
    """Reapply runtime-only checkpointing after every init AND checkpoint reload."""
    verify_stack()
    install_window_guard()
    hf = nlp.get_pipe("transformer").model.transformer
    shim = nlp.get_pipe("transformer").model.layers[0].shims[0]
    mixed = nlp.config["components"]["transformer"]["model"]["mixed_precision"]
    if bool(shim._mixed_precision) != mixed or bool(shim._grad_scaler._enabled) != mixed:
        raise RuntimeError("Reloaded autocast/scaler flags differ from the effective configuration")
    if policy.get("gradient_checkpointing", False):
        if not hf.supports_gradient_checkpointing:
            raise RuntimeError("This encoder does not support gradient checkpointing")
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.config.use_cache = False
        if not hf.is_gradient_checkpointing:
            raise RuntimeError("Gradient checkpointing did not become active")
    nlp._ner_v2_policy = policy
    nlp._ner_v2_diagnostics = diagnostic_path
    return {
        "gradient_checkpointing": bool(hf.is_gradient_checkpointing),
        "checkpoint_use_reentrant": False,
        "all_windows_preserved": True,
        "mixed_precision": mixed,
        "scaler": scaler_state(shim._grad_scaler),
    }


def checked_update(
    nlp,
    batch,
    optimizer,
    *,
    accumulate,
    dropout,
    policy,
    diagnostic_path=None,
    expected_device="cuda",
):
    """Replay bounded AMP overflows; never update either component on bad grads.

    Preflights/full training always use expected_device='cuda'. CPU is exposed
    only for small offline tests of the actual torch/Thinc math and loop wiring.
    """
    import numpy
    import torch
    from spacy.training.loop import subdivide_batch

    hf = nlp.get_pipe("transformer").model.transformer
    shim = nlp.get_pipe("transformer").model.layers[0].shims[0]
    scaler = shim._grad_scaler
    microbatches = list(subdivide_batch(batch, accumulate))
    if not microbatches:
        raise ValueError("Cannot validate an empty batch")
    rng = (
        random.getstate(),
        numpy.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if expected_device == "cuda" else None,
    )
    retry_seed = int.from_bytes(
        hashlib.sha256(rng[2].numpy().tobytes() + rng[1][1].tobytes()).digest()[:4],
        "little",
    )
    attempts = []
    update_index = getattr(nlp, "_ner_v2_update_index", 0)
    nlp._ner_v2_update_index = update_index + 1
    forward_shapes = []
    started = time.perf_counter()

    def forward_check(module, inputs, output):
        value = output.last_hidden_state
        forward_shapes.append(
            {
                "windows": int(value.shape[0]),
                "wordpieces_per_window": int(value.shape[1]),
                "dtype": str(value.dtype),
            }
        )
        # Bound temporary masks to one window at a time, including long records.
        if any(not bool(torch.isfinite(window).all()) for window in value):
            raise NumericalFailure("nonfinite_forward", {"tensor": "last_hidden_state"})

    hook = hf.register_forward_hook(forward_check)
    nlp._ner_v2_in_update = True
    try:
        for attempt in range(policy.get("max_overflow_retries", 8) + 1):
            random.setstate(rng[0])
            numpy.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if expected_device == "cuda":
                import cupy

                torch.cuda.set_rng_state_all(rng[3])
                cupy.random.seed(retry_seed)
            clear_gradients(nlp, hf)
            forward_shapes.clear()
            scale_before = scaler_state(scaler)
            if not numpy.isfinite(scale_before["scale"]) or scale_before["scale"] <= 0:
                raise NumericalFailure("invalid_loss_scale", {"scaler": scale_before})
            proxy = DeferredParameterUnscale(scaler, hf)
            losses = {}
            event = {
                "attempt": attempt,
                "update_index": update_index,
                "stage": getattr(nlp, "_ner_v2_stage", "training"),
                "scale_before": scale_before,
                "microbatch_records": [len(b) for b in microbatches],
                "tokens": sum(len(e) for e in batch),
                "expected_windows": sum(
                    len(group)
                    for group in nlp.get_pipe("transformer").model.attrs["get_spans"](
                        [e.predicted for e in batch]
                    )
                ),
                "records": [
                    {"id": e.reference.user_data.get("record_id"), "tokens": len(e)} for e in batch
                ],
                "replay_seed": retry_seed,
            }
            try:
                shim._grad_scaler = proxy
                for microbatch in microbatches:
                    nlp.update(microbatch, sgd=False, drop=dropout, losses=losses)
                event["scaled_gradients"] = gradient_report(
                    hf, expected_device=expected_device, phase="scaled_before_unscale"
                )
                proxy.finish()
            except Exception as error:
                event.update(
                    outcome="backward_failed",
                    error=repr(error),
                    forward_shapes=list(forward_shapes),
                )
                if isinstance(error, NumericalFailure):
                    event["failure"] = error.diagnostics
                emit(diagnostic_path, event)
                raise
            finally:
                shim._grad_scaler = scaler
            report = gradient_report(
                hf, expected_device=expected_device, phase="unscaled_before_optimizer"
            )
            event.update(
                gradients=report,
                forward_shapes=list(forward_shapes),
                scaler=scaler_state(scaler),
                deferred_backward_calls=proxy.calls,
                losses={k: float(v) for k, v in losses.items()},
                thinc_nonfinite=thinc_report(nlp),
            )
            if sum(shape["windows"] for shape in forward_shapes) != event["expected_windows"]:
                category = "transformer_window_count_mismatch"
            elif report["missing"] or not report["present"]:
                category = "missing_gradient"
            elif report["wrong_device"]:
                category = "gradient_device_mismatch"
            elif any(not numpy.isfinite(v) for v in losses.values()) or event["thinc_nonfinite"]:
                category = "nonfinite_loss_or_unscaled_ner_gradient"
            elif report["nonfinite"] or scaler.found_inf:
                category = (
                    "amp_backward_nonfinite"
                    if scaler._enabled and scaler.found_inf
                    else "unhandled_nonfinite_gradient"
                )
            else:
                category = None
            if category == "amp_backward_nonfinite":
                # Both heads are discarded: do NOT call either finish_update or
                # step_schedules. Same batch/dropout is replayed at a lower scale.
                scaler.update()
                clear_gradients(nlp, hf)
                event.update(outcome="overflow_replayed", scale_after=scaler_state(scaler))
                attempts.append(event)
                emit(diagnostic_path, event)
                if attempt == policy.get("max_overflow_retries", 8):
                    raise NumericalFailure("persistent_amp_nonfinite", {"attempts": attempts})
                if event["scale_after"]["scale"] >= scale_before["scale"]:
                    raise NumericalFailure("scaler_did_not_backoff", {"attempts": attempts})
                continue
            if category:
                event["outcome"] = category
                emit(diagnostic_path, event)
                raise NumericalFailure(category, {"attempts": [*attempts, event]})
            # Validation is after the single unscale and before BOTH optimizers.
            for _, component in nlp.pipeline:
                component.finish_update(optimizer)
            if any(not tensor_finite(p) for p in hf.parameters()) or thinc_report(
                nlp, parameters=True
            ):
                event["outcome"] = "nonfinite_parameter_after_update"
                emit(diagnostic_path, event)
                raise NumericalFailure(
                    "nonfinite_parameter_after_update", {"attempts": [*attempts, event]}
                )
            optimizer.step_schedules()
            event.update(outcome="applied_finite_update", scale_after=scaler_state(scaler))
            attempts.append(event)
            emit(diagnostic_path, event)
            if expected_device == "cuda":
                torch.cuda.synchronize()
            return {
                "seconds": time.perf_counter() - started,
                "tokens": event["tokens"],
                "records": len(batch),
                "losses": event["losses"],
                "optimizer_updates": 1,
                "overflow_retries": attempt,
                "attempts": attempts,
            }
    finally:
        clear_gradients(nlp, hf)
        hook.remove()
        shim._grad_scaler = scaler
        nlp._ner_v2_in_update = False


def install_training_loop(policy, diagnostic_path):
    """Keep spaCy's checkpoint/dev/early-stop machinery; replace only its step.

    Pinned loop semantics below match 3.8.7 (including score ties and step caps).
    Unlike upstream, an AMP overflow replays the whole batch before either head
    updates, and the scheduler advances only after a finite applied update.
    """
    import math

    verify_stack()
    from spacy.training import loop
    from thinc.api import constant

    def guarded(
        nlp,
        optimizer,
        train_data,
        evaluate,
        *,
        dropout,
        eval_frequency,
        accumulate_gradient,
        patience,
        max_steps,
        exclude,
        annotating_components,
        before_update,
    ):
        if exclude or annotating_components:
            raise ValueError("Audited loop supports exactly the transformer -> ner configuration")
        dropouts = constant(dropout) if isinstance(dropout, float) else dropout
        results, losses, words_seen = [], {}, 0
        start = time.perf_counter()
        for step, (epoch, batch) in enumerate(train_data):
            if before_update:
                before_update(nlp, {"step": step, "epoch": epoch})
            update = checked_update(
                nlp,
                batch,
                optimizer,
                accumulate=accumulate_gradient,
                dropout=next(dropouts),
                policy=policy,
                diagnostic_path=diagnostic_path,
            )
            for key, value in update["losses"].items():
                losses[key] = losses.get(key, 0.0) + value
            if step % eval_frequency == 0:
                if optimizer.averages:
                    with nlp.use_params(optimizer.averages):
                        score, scores = evaluate()
                else:
                    score, scores = evaluate()
                if not math.isfinite(score):
                    raise NumericalFailure("nonfinite_development_score", {})
                results.append((score, step))
                is_best = score == max(results)[0]
            else:
                score, scores, is_best = None, None, None
            words_seen += sum(len(e) for e in batch)
            yield (
                batch,
                {
                    "epoch": epoch,
                    "step": step,
                    "score": score,
                    "other_scores": scores,
                    "losses": losses,
                    "checkpoints": results,
                    "seconds": int(time.perf_counter() - start),
                    "words": words_seen,
                    "numerics_overflow_retries": update["overflow_retries"],
                },
                is_best,
            )
            if is_best is not None:
                losses = {}
            best_step = -max((value, -index) for value, index in results)[1]
            if (patience and step - best_step >= patience) or (max_steps and step >= max_steps):
                break

    loop.train_while_improving = guarded

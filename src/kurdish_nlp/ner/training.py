"""Local-only XLM-R training and guarded model selection/test evaluation."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .build import dump, read_jsonl, verify
from .dataset import sha256, spans
from .evaluation import metrics


def gold_doc(nlp, record):
    """Use the same persisted tokenizer as inference; align BIO offsets strictly."""
    text = " ".join(record["tokens"])
    starts, cursor = [], 0
    for word in record["tokens"]:
        starts.append(cursor)
        cursor += len(word) + 1
    doc = nlp.make_doc(text)
    entities = []
    for start, end, label in spans(record["tags"]):
        a, b = starts[start], starts[end - 1] + len(record["tokens"][end - 1])
        entity = doc.char_span(a, b, label=label, alignment_mode="strict")
        if entity is None or entity.text != text[a:b]:
            raise ValueError(f"Unaligned entity in {record['id']}: {(a, b, label)}")
        entities.append(entity)
    doc.ents = entities
    doc.user_data["record_id"] = record["id"]
    doc.user_data["group"] = record["group"]
    return doc


def convert_records(records, path: Path, nlp) -> dict:
    from spacy.tokens import DocBin

    db = DocBin(store_user_data=True)
    counts = Counter()
    for record in records:
        doc = gold_doc(nlp, record)
        counts["segments"] += 1
        counts["bio_tokens"] += len(record["tokens"])
        counts["spacy_tokens"] += len(doc)
        counts["retokenized_segments"] += [t.text for t in doc] != record["tokens"]
        counts["entities"] += len(doc.ents)
        db.add(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(path)
    # Roundtrip catches silent missing spans and sentence splitting.
    loaded = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    if (
        len(loaded) != counts["segments"]
        or sum(len(d.ents) for d in loaded) != counts["entities"]
    ):
        raise ValueError("DocBin roundtrip lost records or entities")
    return dict(counts)


def convert(args):
    import spacy

    verify(args.data, args.lock, args.source)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    nlp = spacy.blank("xx")
    report = {
        split: convert_records(
            read_jsonl(args.data / f"{split}.jsonl"),
            args.output / f"{split}.spacy",
            nlp,
        )
        for split in ("train", "dev", "test")
    }
    dump(
        args.output / "conversion.json",
        {
            "lock_sha256": sha256(args.lock),
            "spacy": spacy.__version__,
            "files": {f.name: sha256(f) for f in args.output.glob("*.spacy")},
            "splits": report,
        },
    )
    print(json.dumps(report, indent=2))


def tree_hash(path: Path) -> dict:
    return {
        str(f.relative_to(path)): sha256(f)
        for f in sorted(path.rglob("*"))
        if f.is_file()
    }


def validate_encoder(encoder: Path) -> dict:
    spec_path = Path("configs/ner/encoder-v2.json")
    spec = json.loads(spec_path.read_text())
    for name, digest in spec["files"].items():
        if sha256(encoder / name) != digest:
            raise ValueError(f"Encoder checksum mismatch: {name}")
    return spec


def wordpiece_check(
    records, nlp, encoder: Path, window: int, stride: int, *, tokenizer=None
) -> dict:
    from spacy_transformers.span_getters import get_strided_spans
    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        str(encoder), local_files_only=True, use_fast=True
    )
    limit = min(512, tokenizer.model_max_length)
    chunks, maximum, count = [], 0, 0

    def flush():
        nonlocal maximum, count
        lengths = tokenizer(
            chunks, truncation=False, padding=False, return_length=True
        )["length"]
        maximum = max(maximum, max(lengths, default=0))
        count += len(lengths)
        if any(n > limit for n in lengths):
            raise ValueError(
                "Transformer window exceeds subword capacity; create a new smaller-window config"
            )
        chunks.clear()

    for record in records:
        doc = gold_doc(nlp, record)
        chunks.extend(span.text for span in get_strided_spans([doc], window, stride)[0])
        if len(chunks) >= 128:
            flush()
    if chunks:
        flush()
    return {
        "windows": count,
        "max_wordpieces": maximum,
        "capacity": limit,
        "truncation": False,
    }


def _run(args, smoke_run: bool):
    import spacy
    import torch
    from spacy.util import load_config

    lock = verify(args.data, args.lock, args.source)
    encoder_spec = validate_encoder(args.encoder)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; prior runs are immutable")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    nlp = spacy.blank("xx")
    records = {s: read_jsonl(args.data / f"{s}.jsonl") for s in ("train", "dev")}
    if smoke_run:
        # Real training annotations only. Cover every legitimate label without test access.
        selected, ids = [], set()
        for label in lock["labels"]:
            candidates = [
                r
                for r in records["train"]
                if any(e[2] == label for e in spans(r["tags"]))
            ]
            r = min(candidates, key=lambda r: (len(r["tokens"]), r["id"]))
            if r["id"] not in ids:
                selected.append(r)
                ids.add(r["id"])
        records["train"] = selected
        records["dev"] = sorted(
            records["dev"], key=lambda r: (len(r["tokens"]), r["id"])
        )[:8]
    cfg = load_config(args.config)
    cfg["components"]["transformer"]["model"]["name"] = str(args.encoder.resolve())
    cfg["paths"]["train"] = str((args.output / "train.spacy").resolve())
    cfg["paths"]["dev"] = str((args.output / "dev.spacy").resolve())
    if smoke_run:
        cfg["training"].update(
            max_steps=1,
            max_epochs=0,
            patience=0,
            eval_frequency=1,
            accumulate_gradient=1,
        )
        cfg["training"]["batcher"]["size"] = 32
        cfg["training"]["optimizer"]["learn_rate"].update(
            warmup_steps=0, total_steps=10
        )
    conversion = {
        s: convert_records(rs, args.output / f"{s}.spacy", nlp)
        for s, rs in records.items()
    }
    window = cfg["components"]["transformer"]["model"]["get_spans"]
    wp = wordpiece_check(
        [r for rs in records.values() for r in rs],
        nlp,
        args.encoder,
        window["window"],
        window["stride"],
    )
    cfg.to_disk(args.output / "config.cfg")
    manifest = {
        "kind": "smoke" if smoke_run else "full",
        "status": "running",
        "seed": cfg["system"]["seed"],
        "dataset_lock_sha256": sha256(args.lock),
        "encoder": encoder_spec,
        "original_config_sha256": sha256(args.config),
        "run_config_sha256": sha256(args.output / "config.cfg"),
        "corpora_sha256": {s: sha256(args.output / f"{s}.spacy") for s in records},
        "conversion": conversion,
        "wordpiece_preflight": wp,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "spacy",
                "spacy-transformers",
                "transformers",
                "torch",
                "numpy",
            )
        },
        "device": args.gpu_id,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "train_ids": [r["id"] for r in records["train"]] if smoke_run else None,
        "dev_ids": [r["id"] for r in records["dev"]] if smoke_run else None,
        "test_accessed": False,
    }
    dump(args.output / "run.json", manifest)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONHASHSEED=str(cfg["system"]["seed"]),
    )
    start = time.perf_counter()
    with (args.output / "train.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "spacy",
                "train",
                str(args.output / "config.cfg"),
                "--output",
                str(args.output / "checkpoints"),
                "--gpu-id",
                str(args.gpu_id),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    manifest.update(
        training_wall_seconds=time.perf_counter() - start,
        status="trained" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        training_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    dump(args.output / "run.json", manifest)
    if result.returncode:
        raise RuntimeError(f"spaCy training failed: see {args.output / 'train.log'}")
    if smoke_run:
        start = time.perf_counter()
        model = args.output / "checkpoints" / "model-best"
        loaded = spacy.load(model)
        if sorted(loaded.get_pipe("ner").labels) != lock["labels"]:
            raise ValueError("Model did not retain all legitimate labels")
        # Use development inputs only for smoke inference and serialization equality.
        texts = [" ".join(r["tokens"]) for r in records["dev"]]
        docs = list(loaded.pipe(texts, batch_size=1))
        output = [[(e.start_char, e.end_char, e.label_) for e in d.ents] for d in docs]
        for text, doc in zip(texts, docs):
            assert doc.text == text
            for e in doc.ents:
                assert text[e.start_char : e.end_char] == e.text
        size = sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
        del loaded, docs
        gc.collect()
        loaded = spacy.load(model)
        again = [
            [(e.start_char, e.end_char, e.label_) for e in d.ents]
            for d in loaded.pipe(texts, batch_size=1)
        ]
        if again != output:
            raise ValueError("Reload changed inference")
        manifest.update(
            status="smoke_validated",
            model_size_bytes=size,
            reload_and_inference_seconds=time.perf_counter() - start,
            labels=lock["labels"],
            integration_approved=False,
            note="Two-step engineering smoke, not a trained quality baseline or test-set result.",
        )
        dump(args.output / "run.json", manifest)
    print(json.dumps(manifest, indent=2))


def smoke(args):
    _run(args, True)


def train(args):
    _run(args, False)


def benchmark(args):
    """Preflight all training/development windows, then time real CPU work.

    Updates use training annotations and stay in memory; saved models and the
    frozen test partition are never changed or predicted on.
    """
    import math

    import spacy
    import torch
    from spacy.training import Example
    from spacy.util import fix_random_seed

    from .contract import entities_payload

    verify(args.data, args.lock, args.source)
    validate_encoder(args.encoder)
    if args.output.exists():
        raise FileExistsError(args.output)
    fix_random_seed(1729)
    torch.set_num_threads(4)
    nlp = spacy.load(args.model)
    records = {s: read_jsonl(args.data / f"{s}.jsonl") for s in ("train", "dev")}
    window = nlp.config["components"]["transformer"]["model"]["get_spans"]
    wp = wordpiece_check(
        [r for rs in records.values() for r in rs],
        nlp,
        args.encoder,
        window["window"],
        window["stride"],
    )
    # Fixed ordinary-length segments give a useful timing range; not quality metrics.
    dev = [r for r in records["dev"] if 24 <= len(r["tokens"]) <= 32][:32]
    texts = [" ".join(r["tokens"]) for r in dev]
    nlp(texts[0])
    start = time.perf_counter()
    docs = list(nlp.pipe(texts, batch_size=1))
    seconds = time.perf_counter() - start
    for doc in docs:
        entities_payload(
            doc, "sorani-ner-v2-smoke", "0.0.0-smoke", list(nlp.get_pipe("ner").labels)
        )
    # Include a real astral input to verify code point offset invariants.
    entities_payload(
        nlp("😀 " + texts[0]),
        "sorani-ner-v2-smoke",
        "0.0.0-smoke",
        list(nlp.get_pipe("ner").labels),
    )
    selected = [r for r in records["train"] if 24 <= len(r["tokens"]) <= 32][:2]
    optimizer = nlp.resume_training()
    updates = []
    for record in selected:
        reference = gold_doc(nlp, record)
        example = Example(nlp.make_doc(reference.text), reference)
        start = time.perf_counter()
        losses = {}
        nlp.update([example], sgd=optimizer, drop=0.1, losses=losses)
        elapsed = time.perf_counter() - start
        if not all(math.isfinite(v) for v in losses.values()):
            raise ValueError("Non-finite transformer/NER loss")
        updates.append(
            {
                "record": record["id"],
                "tokens": len(reference),
                "seconds": elapsed,
                "losses": {name: float(value) for name, value in losses.items()},
            }
        )
    token_rate = sum(u["tokens"] for u in updates) / sum(u["seconds"] for u in updates)
    train_tokens = sum(len(r["tokens"]) for r in records["train"])
    report = {
        "dataset_lock_sha256": sha256(args.lock),
        "model_files": tree_hash(args.model),
        "wordpiece_preflight_train_dev": wp,
        "test_predicted": False,
        "inference": {
            "segments": len(docs),
            "tokens": sum(len(d) for d in docs),
            "seconds": seconds,
            "segments_per_second": len(docs) / seconds,
            "tokens_per_second": sum(len(d) for d in docs) / seconds,
            "batch_size": 1,
            "device": "cpu",
            "threads": 4,
        },
        "training_updates": updates,
        "training_tokens_per_second": token_rate,
        "estimated_15_epoch_compute_hours": 15 * train_tokens / token_rate / 3600,
        "estimate_caveat": "Two ordinary-length single-record CPU updates only. Full run batching, long segments, evaluation, checkpoints, memory pressure and early stopping change runtime; not an ETA.",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "contract_validated": True,
        "checkpoint_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, report)
    print(json.dumps(report, indent=2))


def select(args):
    """Freeze a completed full run's dev-selected checkpoint before test access."""
    lock = verify(args.data, args.lock, args.source)
    if args.output.exists():
        raise FileExistsError(args.output)
    run_root = args.model.parent.parent
    run = json.loads((run_root / "run.json").read_text())
    if (
        run["kind"] != "full"
        or run["status"] != "trained"
        or run["dataset_lock_sha256"] != sha256(args.lock)
    ):
        raise ValueError(
            "Selection requires a completed full run on this frozen dataset"
        )
    if args.model.resolve() != (run_root / "checkpoints" / "model-best").resolve():
        raise ValueError("Select the checkpoint chosen by development F1")
    meta = json.loads((args.model / "meta.json").read_text())
    if sorted(meta["labels"]["ner"]) != lock["labels"]:
        raise ValueError("Model label inventory differs from frozen data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(
        args.output,
        {
            "dataset_lock_sha256": sha256(args.lock),
            "model_files": tree_hash(args.model),
            "run_sha256": sha256(run_root / "run.json"),
            "dev_performance": meta.get("performance"),
            "criterion": "spaCy best development entity micro F1",
            "test_evaluated": False,
        },
    )


def evaluate(args):
    import spacy
    import torch

    lock = verify(args.data, args.lock, args.source)
    selection = json.loads(args.selection.read_text())
    if (
        selection["dataset_lock_sha256"] != sha256(args.lock)
        or tree_hash(args.model) != selection["model_files"]
    ):
        raise ValueError("Model or dataset changed after selection")
    if (
        args.output.exists()
        or (args.selection.parent / "test-evaluation.receipt.json").exists()
    ):
        raise FileExistsError(
            "Test already evaluated or output exists; do not tune on repeated test evaluations"
        )
    torch.set_num_threads(4)
    nlp = spacy.load(args.model)
    records = read_jsonl(args.data / "test.jsonl")
    texts = [" ".join(r["tokens"]) for r in records]
    references = [gold_doc(nlp, r) for r in records]
    # Warmup excluded from measured inference time.
    nlp(texts[0])
    start = time.perf_counter()
    docs = list(nlp.pipe(texts, batch_size=1))
    elapsed = time.perf_counter() - start
    gold = [{(e.start_char, e.end_char, e.label_) for e in d.ents} for d in references]
    predicted = [{(e.start_char, e.end_char, e.label_) for e in d.ents} for d in docs]
    report = metrics(gold, predicted, lock["labels"])
    support = json.loads((args.data / "audit.json").read_text())["insufficient_support"]
    report.update(
        dataset_lock_sha256=sha256(args.lock),
        selection_sha256=sha256(args.selection),
        evaluation_scope=lock["evaluation_scope"],
        insufficient_support={label: report["per_label"][label] for label in support},
        model_size_bytes=sum(
            f.stat().st_size for f in args.model.rglob("*") if f.is_file()
        ),
        inference={
            "seconds": elapsed,
            "segments_per_second": len(docs) / elapsed,
            "tokens_per_second": sum(len(d) for d in docs) / elapsed,
            "batch_size": 1,
            "device": "cpu",
            "threads": 4,
            "platform": platform.platform(),
        },
        dataset_limitations=json.loads((args.data / "audit.json").read_text())[
            "limitations"
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, report)
    dump(
        args.selection.parent / "test-evaluation.receipt.json",
        {
            "selection_sha256": sha256(args.selection),
            "report_sha256": sha256(args.output),
        },
    )

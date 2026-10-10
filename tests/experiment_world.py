"""Synthetic, offline stand-in for a frozen dataset build and UD benchmark.

``make_world`` writes train/dev/test JSONL, provenance, review queue, manifest,
lock, build config and a small UD-style benchmark with correct hashes, plus an
experiment configuration with tiny search grids. Every text is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.build.config import canonical_json
from kurdish_nlp.langid.normalization import normalize_text

LABELS = ("ckb", "kmr", "sdh", "ar", "fa", "tr", "en")
WORDS = {
    "ckb": "ئەمڕۆ کەشوهەوا زۆر خۆشە من کتێبێکی نوێم کڕی باوکم لە بازاڕ کار دەکات "
    "ئێمە بۆ گەشت دەچین خوشکەکەم مامۆستایە شار جەنجاڵە منداڵەکان باخچە یاری دەکەن",
    "sdh": "هەر سوو چم ئەڕا مەدرەسە ئیمڕوو هەوا سەردە کتاوەگەی منە براگەم کەێد "
    "کوورەسە دالگم شیو مەپەزێد ڕووخانە پڕ ئاوە تێەم دێ دوورە ئێمە خواردیم",
    "kmr": "ez ê sibê biçim bajêr û hevalên xwe bibînim pirtûka min li ser maseya "
    "mezin zarok baxçeyê dibistanê bi kêfxweşî dilîzin bavê her roj diçe kar",
    "ar": "ذهب الولد إلى المدرسة صباحا الكتاب على الطاولة الكبيرة نحن نحب السفر "
    "البحر السماء زرقاء والجو جميل شربت القهوة مع أصدقائي",
    "fa": "امروز به کتابخانه رفتم خواهرم نقاشی را دوست دارد ما در تابستان سفر "
    "رفتیم این شهر بسیار زیبا است باران دیشب خیلی شدید بود",
    "tr": "yeni diller öğrenmeyi seviyorum öğretmen yavaşça pazara yürüdü bugün "
    "hava çok güzel kitap masanın üstünde çocuklar bahçede oynuyor yarın",
    "en": "the teacher walked slowly to market weather is nice today book on "
    "table children play garden i will come tomorrow river full of water",
}
COUNTS = {"train": 40, "dev": 16, "test": 12}
SOURCES = {
    "ckb": ("tatoeba", "wikimedia-ckb"),
    "kmr": ("tatoeba", "wikimedia-kmr"),
    "sdh": ("parme", "tatoeba"),
    "ar": ("tatoeba",),
    "fa": ("tatoeba",),
    "tr": ("tatoeba",),
    "en": ("tatoeba",),
}
UD_STRICT = [
    "Le bawuşî elan xalî key we naw sebedege.",
    "Ême çîme bazar û nan xwardîme.",
    "Min wit ke em kitêbe xasa.",
    "Ew kese hate mal û danîşt.",
    "Hewa emro serd bî.",
    "Kur û kinaçe le bax bîn.",
]
UD_EXPLORATORY = [
    "Ez ê sibê biçim bajarê mezin.",
    "Zarokên me li dibistanê dixwînin.",
    "Ev pirtûk li ser dîroka Kurdistanê ye.",
    "Hevalê min her roj tê mala me.",
    "Baran li çiyayan dibare.",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _sentences(rng: random.Random, label: str, count: int, seen: set[str]) -> list[str]:
    words = WORDS[label].split()
    result = []
    while len(result) < count:
        text = normalize_text(
            " ".join(rng.choice(words) for _ in range(rng.randint(3, 7)))
        )
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def make_world(root: Path, *, seed: int = 7, **config_overrides: Any) -> Path:
    """Write the synthetic world under ``root`` and return the experiment config."""

    rng = random.Random(seed)
    build = root / "data/builds/world-v1"
    bench = root / "data/benchmarks/ud"
    build.mkdir(parents=True)
    bench.mkdir(parents=True)
    build_config = {"dataset_name": "world-v1", "note": "synthetic test build"}
    build_config_path = root / "configs/world-v1.json"
    build_config_path.parent.mkdir(parents=True)
    build_config_path.write_text(json.dumps(build_config), encoding="utf-8")
    build_sha = hashlib.sha256(canonical_json(build_config).encode()).hexdigest()

    seen: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in COUNTS}
    provenance, review = [], []
    group_counter = 0
    for label in LABELS:
        for split, count in COUNTS.items():
            texts = _sentences(rng, label, count, seen)
            for i, text in enumerate(texts):
                # Pairs of consecutive records share a document group.
                if i % 2 == 0:
                    group_counter += 1
                source = SOURCES[label][i % len(SOURCES[label])]
                record_id = f"{source}:{label}:{split}:{i:03d}"
                splits[split].append(
                    {
                        "id": record_id,
                        "text": text,
                        "label": label,
                        "source": f"{source}:snapshot",
                        "domain": "encyclopedia"
                        if source.startswith("wikimedia")
                        else "translated_prompt"
                        if source == "parme"
                        else "conversational_translation",
                        "document_id": f"world-v1:group:{group_counter:05d}",
                        "license": "test-only",
                        "split": split,
                    }
                )
                contributor = None if source.startswith("wikimedia") else f"user{i % 4}"
                provenance.append(
                    {
                        "canonical_record_id": record_id,
                        "source_id": source,
                        "contributor_id": contributor if source == "tatoeba" else None,
                        "translator_id": contributor if source == "parme" else None,
                        "language_variety": "Kirmashani" if source == "parme" else None,
                        "transformations": [
                            {
                                "name": "dataset_build",
                                "parameters": {"grouping": {"giant_group": False}},
                            }
                        ],
                    }
                )
                if i == 0:
                    review.append(
                        {
                            "record_id": record_id,
                            "flags": ["mixed_script"],
                            "label": label,
                        }
                    )
    for split, rows in splits.items():
        rows.sort(key=lambda row: row["id"])
        _jsonl(build / f"{split}.jsonl", rows)
    _jsonl(build / "provenance.jsonl", provenance)
    _jsonl(build / "review_queue.jsonl", review)
    counts = {
        label: {
            split: sum(row["label"] == label for row in rows)
            for split, rows in splits.items()
        }
        for label in LABELS
    }
    names = (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "provenance.jsonl",
        "review_queue.jsonl",
    )
    hashes = {name: sha256_file(build / name) for name in names}
    lock_path = root / "configs/world-v1.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "dataset": "world-v1",
                "dataset_version": "1.0.0",
                "config_sha256": build_sha,
                "counts": counts,
                "outputs": hashes,
            }
        ),
        encoding="utf-8",
    )
    (build / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "world-v1",
                "dataset_version": "1.0.0",
                "config": {"sha256": build_sha},
            }
        ),
        encoding="utf-8",
    )

    ud_rows = []
    for i, text in enumerate(UD_STRICT):
        ud_rows.append(_ud_row(f"ud:sdh_garrusi:test:s{i}", "sdh", text, True, []))
    for i, text in enumerate(UD_EXPLORATORY):
        ud_rows.append(
            _ud_row(
                f"ud:kmr_kurmanji:test:k{i}",
                "kmr",
                text,
                False,
                ["legacy_ud_prior_repository_exposure"],
            )
        )
    ud_rows.append(
        _ud_row(
            "ud:sdh_garrusi:test:dup",
            "sdh",
            "Dup text here.",
            False,
            ["within_ud_duplicate_text"],
        )
    )
    ud_rows.append(
        _ud_row(
            "ud:kmr_kurmanji:test:ov",
            "kmr",
            "Ev hevok overlap e.",
            False,
            ["legacy_ud_prior_repository_exposure", "canonical_source_overlap"],
        )
    )
    _jsonl(bench / "ud.jsonl", ud_rows)
    (bench / "ud.manifest.json").write_text(
        json.dumps({"benchmark": "ud-world"}), encoding="utf-8"
    )

    config = {
        "schema_version": 1,
        "experiment_id": "world-exp",
        "experiment_version": "0.0.1",
        "description": "synthetic integration test",
        "seed": 42,
        "labels": list(LABELS),
        "focus_labels": ["ckb", "kmr", "sdh"],
        "abstention_label": "und",
        "dataset": {
            "name": "world-v1",
            "version": "1.0.0",
            "build_dir": str(build),
            "build_config": str(build_config_path),
            "lock_file": str(lock_path),
            "config_sha256": build_sha,
            "expected_counts": {s: len(rows) for s, rows in splits.items()},
            "expected_sha256": hashes,
        },
        "development_protocol": {
            "method": "group_disjoint_greedy_stratified_v1",
            "group_field": "document_id",
            "strata": ["label", "source_family"],
            "roles": {"dev_tune": 0.5, "dev_calib": 0.5},
            "calibration_crossfit_folds": 2,
        },
        "external_benchmark": {
            "benchmark_id": "ud-world",
            "records": {
                "path": str(bench / "ud.jsonl"),
                "sha256": sha256_file(bench / "ud.jsonl"),
            },
            "manifest": {
                "path": str(bench / "ud.manifest.json"),
                "sha256": sha256_file(bench / "ud.manifest.json"),
            },
            "strict": {
                "name": "ud_strict_garrusi",
                "title": "STRICT synthetic Garrusi",
                "expected_language": "sdh",
                "expected_count": len(UD_STRICT),
            },
            "exploratory": {
                "name": "ud_exploratory_kurmanji",
                "title": "EXPLORATORY — prior repository exposure: synthetic Kurmanji",
                "expected_language": "kmr",
                "expected_count": len(UD_EXPLORATORY),
                "allowed_exclusion_reasons": ["legacy_ud_prior_repository_exposure"],
            },
            "restrictions": ["no_training", "no_tuning"],
        },
        "models": {
            "fasttext": {
                "artifact": "model.bin",
                "fixed": {
                    "loss": "softmax",
                    "minCount": 1,
                    "minCountLabel": 1,
                    "bucket": 100000,
                    "thread": 1,
                    "verbose": 0,
                    "lr": 0.5,
                },
                "grid": {
                    "subwords": [[1, 3]],
                    "wordNgrams": [1],
                    "dim": [8],
                    "epoch": [5, 15],
                },
            },
            "tfidf_lr": {
                "artifact": "model.joblib",
                "vectorizer_fixed": {
                    "analyzer": "char",
                    "sublinear_tf": True,
                    "lowercase": False,
                    "strip_accents": None,
                    "dtype": "float32",
                },
                "vectorizer_grid": {
                    "ngram_range": [[1, 3]],
                    "min_df": [1],
                    "max_features": [5000],
                },
                "classifier_fixed": {"solver": "lbfgs", "max_iter": 300, "tol": 0.0001},
                "classifier_grid": {"C": [1.0, 10.0], "class_weight": [None]},
            },
        },
        "selection": {
            "role": "dev_tune",
            "metric": "macro_f1",
            "tie_breakers": ["log_loss"],
        },
        "calibration": {
            "method": "temperature_scaling",
            "fit_role": "dev_calib",
            "temperature_bounds": [0.05, 20.0],
            "probability_floor": 1e-12,
            "ece_bins": 10,
        },
        "abstention": {
            "fit_role": "dev_calib",
            "minimum_letter_count": 3,
            "reference_policy": {"confidence_threshold": 0.7, "margin_threshold": 0.2},
            "confidence_grid": [0.0, 0.5, 0.9],
            "margin_grid": [0.0, 0.2],
            "target_selective_accuracy": 0.99,
            "minimum_coverage": 0.5,
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "length_buckets": [
                {"name": "01-05", "min_tokens": 1, "max_tokens": 5},
                {"name": "06+", "min_tokens": 6, "max_tokens": None},
            ],
            "min_slice_support": 5,
            "contributor_tiers": {"top": 1, "head": 2},
            "bootstrap": {"resamples": 30, "confidence": 0.95},
            "latency": {"single_text_samples": 10, "warmup": 2},
            "errors": {
                "focus_pairs": [["ckb", "sdh"], ["sdh", "ckb"], ["kmr", "tr"]],
                "max_per_pair": 5,
                "max_low_confidence_per_label": 3,
                "markdown_examples_per_pair": 2,
            },
        },
        "resources": {
            "max_threads": 1,
            "memory_budget_bytes": 4_000_000_000,
            "max_candidates": {"fasttext": 4, "tfidf_lr": 4},
        },
        "artifacts": {"root": str(root / "artifacts/world-exp")},
    }
    for dotted, value in config_overrides.items():
        target = config
        *parents, last = dotted.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    path = root / "configs/world-exp.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _ud_row(record_id, language, text, strict, reasons) -> dict[str, Any]:
    treebank = "sdh_garrusi" if language == "sdh" else "kmr_kurmanji"
    return {
        "benchmark": "ud-world",
        "evaluation_partition": "external_test",
        "exclusion_reasons": reasons,
        "expected_language": language,
        "id": record_id,
        "license": "test-only",
        "original_split": "test",
        "slice": f"{treebank}_synthetic",
        "source": "universal-dependencies",
        "strict_external": strict,
        "text": text,
        "treebank": treebank,
    }

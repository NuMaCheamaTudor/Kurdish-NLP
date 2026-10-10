"""Comparison reports, slice tables, Kurdish error analysis and figures.

Reports are a pure function of frozen inputs (evaluation files, stored
predictions, runtime files and the verified dataset), written with sorted keys,
fixed rounding and no timestamps, so regenerating them yields identical bytes.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kurdish_nlp.langid.build.export import write_json, write_jsonl
from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.experiments.calibration import apply_temperature
from kurdish_nlp.langid.experiments.config import ExperimentConfig
from kurdish_nlp.langid.experiments.data import (
    Example,
    ExampleSet,
    LoadedData,
    load_external_benchmark,
)
from kurdish_nlp.langid.experiments.evaluation import (
    group_bootstrap,
    policy_decisions,
    rnd,
    top_two,
)
from kurdish_nlp.langid.experiments.runner import (
    DISPLAY,
    FAMILIES,
    Paths,
    argmax,
    read_predictions,
)
from kurdish_nlp.langid.experiments.slices import REPORT_TABLES, slice_rows
from kurdish_nlp.langid.reliability import ReliabilityPolicy

# Reference palette (dataviz skill): categorical slots 1-2, sequential blue ramp.
SERIES = {"fasttext": "#2a78d6", "tfidf_lr": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = [
    "#fcfcfb",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class FamilyView:
    """Frozen outputs of one model family needed by the reports."""

    def __init__(self, config: ExperimentConfig, paths: Paths, family: str, lock: dict):
        directory = paths.model_dir(family)
        self.family = family
        self.frozen = lock["families"][family]
        self.evaluation = _load_json(directory / "evaluation.json")
        self.training = _load_json(directory / "training.json")
        self.calibration = _load_json(directory / "calibration.json")
        self.runtime = _load_json(directory / "runtime.json")
        external = directory / "evaluation_external.json"
        self.external = _load_json(external) if external.is_file() else None
        self.config = config
        self.paths = paths

    def scores(self, examples: ExampleSet) -> tuple[np.ndarray, np.ndarray]:
        raw = read_predictions(
            self.paths.predictions(self.family, examples.name), examples
        )
        temperature = float(self.frozen["temperature"])
        calibrated = (
            apply_temperature(
                raw, temperature, self.config.calibration.probability_floor
            )
            if temperature != 1.0
            else raw
        )
        return raw, calibrated

    def decisions(self, examples: ExampleSet, calibrated: np.ndarray):
        return policy_decisions(
            ReliabilityPolicy(**self.frozen["policy"]),
            examples.texts,
            calibrated,
            self.config.label_values,
        )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key)) for key in columns})
    return buffer.getvalue()


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def per_class_rows(views: dict[str, FamilyView]) -> list[dict[str, Any]]:
    rows = []
    for family, view in views.items():
        for role, result in view.evaluation["roles"].items():
            for label, values in result["classification"]["per_class"].items():
                tuned = result["abstention"]["tuned_policy_calibrated"]["per_class"][
                    label
                ]
                rows.append(
                    {
                        "model": family,
                        "role": role,
                        "label": label,
                        **values,
                        "tuned_coverage": tuned["coverage"],
                        "tuned_selective_recall": tuned["selective_recall"],
                    }
                )
    return rows


def slice_tables(
    config: ExperimentConfig,
    views: dict[str, FamilyView],
    test: ExampleSet,
) -> dict[str, list[dict[str, Any]]]:
    labels = config.label_values
    y = np.asarray(test.label_indices(config))
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family, view in views.items():
        _, calibrated = view.scores(test)
        pred = argmax(calibrated)
        accepted = np.asarray(
            [lang != "und" for lang, _ in view.decisions(test, calibrated)]
        )
        for stem, dimension, by_language in REPORT_TABLES:
            for row in slice_rows(
                test.examples,
                y,
                pred,
                labels,
                dimension,
                by_language=by_language,
                min_support=config.evaluation.min_slice_support,
                accepted=accepted,
            ):
                tables[stem].append({"model": family, "role": "test", **row})
    return dict(tables)


SLICE_COLUMNS = [
    "model",
    "role",
    "dimension",
    "slice",
    "language",
    "support",
    "correct",
    "accuracy",
    "macro_f1_present_labels",
    "labels_present",
    "top_wrong_prediction",
    "top_wrong_count",
    "coverage",
    "selective_accuracy",
    "low_support",
]
CLASS_COLUMNS = [
    "model",
    "role",
    "label",
    "precision",
    "recall",
    "f1",
    "support",
    "predicted",
    "tuned_coverage",
    "tuned_selective_recall",
]


# --------------------------------------------------------------------------
# Kurdish confusion and error records
# --------------------------------------------------------------------------


def _error_record(
    example: Example,
    family: str,
    labels: tuple[str, ...],
    raw_row: np.ndarray,
    cal_row: np.ndarray,
    decision: tuple[str, str | None],
    other_prediction: str,
    category: str,
) -> dict[str, Any]:
    order = np.argsort(-cal_row, kind="stable")
    top1, top2 = int(order[0]), int(order[1])
    predicted = labels[top1]
    return {
        "record_id": example.id,
        "model": family,
        "role": example.role,
        "category": category,
        "true_label": example.label.value,
        "predicted_label": predicted,
        "top1_confidence": rnd(cal_row[top1]),
        "top1_raw_score": rnd(raw_row[top1]),
        "top2_label": labels[top2],
        "top2_confidence": rnd(cal_row[top2]),
        "margin": rnd(cal_row[top1] - cal_row[top2]),
        "true_label_confidence": rnd(cal_row[labels.index(example.label.value)]),
        "abstained": decision[0] == "und",
        "abstention_reason": decision[1],
        "source": example.source_family,
        "domain": example.domain,
        "tokens": example.tokens,
        "length_bucket": example.length_bucket,
        "script": example.script,
        "contributor_tier": example.contributor_tier,
        "variety": example.variety,
        "review_flags": list(example.review_flags),
        "other_model_prediction": other_prediction,
        "both_models_agree_on_this_label": other_prediction == predicted,
        "text": example.text,
    }


def kurdish_analysis(
    config: ExperimentConfig,
    views: dict[str, FamilyView],
    test: ExampleSet,
    strict: ExampleSet | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = config.label_values
    focus = {label.value for label in config.focus_labels}
    spec = config.evaluation.errors
    y = np.asarray(test.label_indices(config))
    data: dict[str, dict[str, Any]] = {}
    for family, view in views.items():
        raw, cal = view.scores(test)
        data[family] = {
            "raw": raw,
            "cal": cal,
            "pred": argmax(cal),
            "decisions": view.decisions(test, cal),
        }
    analysis: dict[str, Any] = {
        "role": "test",
        "note": "Disagreements are measured against unreviewed source labels; some "
        "may be label errors. 'both models agree' marks candidates for label review.",
        "models": {},
    }
    errors: list[dict[str, Any]] = []
    families = list(views)
    for family in families:
        other = (
            next(f for f in families if f != family) if len(families) > 1 else family
        )
        current = data[family]
        pred, cal = current["pred"], current["cal"]
        _, _, top1, _ = top_two(cal)
        rows: dict[str, Any] = {}
        for label in sorted(focus):
            index = labels.index(label)
            members = y == index
            rows[label] = {
                "support": int(members.sum()),
                "predicted_as": {
                    labels[j]: int((members & (pred == j)).sum())
                    for j in range(len(labels))
                },
            }
        pairs = {}
        for true_label, predicted_label in spec.focus_pairs:
            t, p = labels.index(true_label.value), labels.index(predicted_label.value)
            mask = (y == t) & (pred == p)
            idx = np.flatnonzero(mask)
            members = [test.examples[i] for i in idx]
            abstained = sum(current["decisions"][i][0] == "und" for i in idx)
            pairs[f"{true_label.value}->{predicted_label.value}"] = {
                "count": int(mask.sum()),
                "rate_of_true_label": rnd(mask.sum() / max(1, (y == t).sum())),
                "mean_confidence": rnd(top1[idx].mean()) if len(idx) else None,
                "abstained_under_tuned_policy": int(abstained),
                "by_length": dict(
                    sorted(Counter(e.length_bucket for e in members).items())
                ),
                "by_source": dict(
                    sorted(Counter(e.source_family for e in members).items())
                ),
                "by_script": dict(sorted(Counter(e.script for e in members).items())),
                "by_domain": dict(sorted(Counter(e.domain for e in members).items())),
                "also_wrong_same_way_in_other_model": int(
                    (data[other]["pred"][idx] == p).sum()
                ),
            }
        analysis["models"][family] = {"kurdish_rows": rows, "focus_pairs": pairs}

        ids = [example.id for example in test.examples]
        # Most confident errors first; record ID breaks ties deterministically.
        confident_first = sorted(
            range(len(ids)), key=lambda i: (-float(top1[i]), ids[i])
        )
        least_confident_first = confident_first[::-1]
        selected: list[tuple[int, str]] = []
        listed: set[int] = set()
        for true_label, predicted_label in spec.focus_pairs:
            t, p = labels.index(true_label.value), labels.index(predicted_label.value)
            members = [i for i in confident_first if y[i] == t and pred[i] == p]
            listed.update(members)
            category = f"{true_label.value}->{predicted_label.value}"
            selected.extend((i, category) for i in members[: spec.max_per_pair])
        other_errors: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i in confident_first:
            if pred[i] == y[i] or i in listed:
                continue
            if labels[y[i]] in focus or labels[pred[i]] in focus:
                other_errors[(labels[y[i]], labels[pred[i]])].append(i)
        for (t_label, p_label), members in sorted(other_errors.items()):
            category = f"other:{t_label}->{p_label}"
            selected.extend((i, category) for i in members[: spec.max_per_pair])
        # "Low confidence": correct, but calibrated top-1 below max(threshold, 0.9).
        ceiling = max(
            float(views[family].frozen["policy"]["confidence_threshold"]), 0.9
        )
        for label in sorted(focus):
            index = labels.index(label)
            members = [
                i
                for i in least_confident_first
                if y[i] == index and pred[i] == y[i] and top1[i] < ceiling
            ]
            category = f"low_confidence_correct:{label}"
            selected.extend(
                (i, category) for i in members[: spec.max_low_confidence_per_label]
            )
        other_pred = data[other]["pred"]
        for i, category in selected:
            errors.append(
                _error_record(
                    test.examples[i],
                    family,
                    labels,
                    current["raw"][i],
                    cal[i],
                    current["decisions"][i],
                    labels[int(other_pred[i])],
                    category,
                )
            )
    if strict is not None:
        _strict_errors(config, views, strict, errors)
    analysis["error_overlap"] = _error_overlap(labels, focus, y, data)
    return analysis, errors


def _strict_errors(config, views, strict: ExampleSet, errors: list) -> None:
    labels = config.label_values
    families = list(views)
    preds = {}
    scored = {}
    for family, view in views.items():
        raw, cal = view.scores(strict)
        scored[family] = (raw, cal, view.decisions(strict, cal))
        preds[family] = argmax(cal)
    y = np.asarray(strict.label_indices(config))
    for family in families:
        other = (
            next(f for f in families if f != family) if len(families) > 1 else family
        )
        raw, cal, decisions = scored[family]
        _, _, top1, _ = top_two(cal)
        by_pred = defaultdict(list)
        for i in np.flatnonzero(preds[family] != y):
            by_pred[labels[preds[family][i]]].append(int(i))
        for predicted, idx in sorted(by_pred.items()):
            chosen = sorted(idx, key=lambda i: (-float(top1[i]), strict.examples[i].id))
            for i in chosen[: config.evaluation.errors.max_per_pair]:
                errors.append(
                    _error_record(
                        strict.examples[i],
                        family,
                        labels,
                        raw[i],
                        cal[i],
                        decisions[i],
                        labels[int(preds[other][i])],
                        f"ud_strict:sdh->{predicted}",
                    )
                )


def _error_overlap(labels, focus, y, data) -> dict[str, Any]:
    families = sorted(data)
    if len(families) != 2:
        return {}
    a, b = families
    kurdish = np.isin(y, [labels.index(label) for label in sorted(focus)])
    wrong_a = data[a]["pred"] != y
    wrong_b = data[b]["pred"] != y
    return {
        "scope": "test records whose true label is ckb, kmr or sdh",
        "records": int(kurdish.sum()),
        "both_wrong": int((kurdish & wrong_a & wrong_b).sum()),
        "both_wrong_same_label": int(
            (kurdish & wrong_a & wrong_b & (data[a]["pred"] == data[b]["pred"])).sum()
        ),
        f"only_{a}_wrong": int((kurdish & wrong_a & ~wrong_b).sum()),
        f"only_{b}_wrong": int((kurdish & ~wrong_a & wrong_b).sum()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": False,
            "font.size": 10,
            "svg.hashsalt": "kurdish-nlp",
        }
    )
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": None}, facecolor=SURFACE)


def confusion_figure(path: Path, family: str, matrix: list[list[int]], labels) -> None:
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    plt = _pyplot()
    counts = np.asarray(matrix, dtype=np.float64)
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(rates, cmap=cmap, norm=PowerNorm(gamma=0.4, vmin=0, vmax=1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(counts[i, j])
            if value == 0:
                continue
            dark = rates[i, j] ** 0.4 > 0.62
            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if dark else INK,
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"{DISPLAY[family]}: internal test confusion\n"
        "cell colour = row share (power scale), number = count",
        fontsize=10,
        loc="left",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("share of true-label row")
    bar.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def risk_coverage_figure(path: Path, views: dict[str, FamilyView]) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    notes = []
    for family, view in views.items():
        test = view.evaluation["roles"]["test"]
        curve = test["risk_coverage"]["calibrated"]["curve"]
        x = [point["coverage"] for point in curve]
        y = [point["risk"] * 100 for point in curve]
        ax.plot(x, y, color=SERIES[family], linewidth=2, label=DISPLAY[family])
        for name, marker in (
            ("tuned_policy_calibrated", "o"),
            ("reference_policy_raw", "D"),
        ):
            point = test["abstention"][name]
            risk = (1 - point["selective_accuracy"]) * 100
            ax.plot(
                [point["coverage"]],
                [risk],
                marker=marker,
                markersize=8,
                color=SERIES[family],
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                linestyle="none",
            )
            kind = (
                "tuned (calibrated)" if name.startswith("tuned") else "0.70/0.20 (raw)"
            )
            notes.append(
                f"{DISPLAY[family]}, {kind}: coverage {point['coverage']:.3f}, "
                f"risk {risk:.2f}%"
            )
    ax.plot([], [], marker="o", color=INK_2, linestyle="none", label="tuned policy")
    ax.plot([], [], marker="D", color=INK_2, linestyle="none", label="0.70/0.20 policy")
    ax.text(
        0.02,
        0.62,
        "\n".join(notes),
        transform=ax.transAxes,
        fontsize=8,
        color=INK_2,
        va="top",
    )
    ax.set_xlabel("Coverage (share of test records answered)")
    ax.set_ylabel("Selective risk (% errors among answered)")
    ax.set_title(
        "Internal test: selective risk vs coverage (calibrated top-1 score)\n"
        "markers: frozen abstention policies; the dev_calib-tuned one abstains on nothing",
        fontsize=10,
        loc="left",
    )
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_xlim(0.8, 1.005)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def reliability_figure(path: Path, views: dict[str, FamilyView]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(views), figsize=(4.2 * len(views), 4.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (family, view) in zip(axes, views.items(), strict=True):
        probabilities = view.evaluation["roles"]["test"]["probabilities"]
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
        for kind, color, style in (
            ("raw", INK_2, "--"),
            ("calibrated", SERIES[family], "-"),
        ):
            table = [
                b for b in probabilities[kind]["reliability_table"] if b["count"] > 0
            ]
            ax.plot(
                [b["mean_confidence"] for b in table],
                [b["accuracy"] for b in table],
                color=color,
                linestyle=style,
                linewidth=2,
                marker="o",
                markersize=5,
                label=f"{kind} (ECE {probabilities[kind]['ece']:.4f})",
            )
        calibrated = probabilities["calibrated"]["reliability_table"]
        total = sum(b["count"] for b in calibrated)
        top_share = calibrated[-1]["count"] / total if total else 0.0
        below_half = sum(b["count"] for b in calibrated if b["upper"] <= 0.6)
        ax.set_title(
            f"{DISPLAY[family]}: test reliability\n"
            f"top bin holds {top_share:.1%} of records; bins <= 0.6 hold {below_half}",
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("Mean top-1 score in bin")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Accuracy in bin")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Comparison summary and Markdown
# --------------------------------------------------------------------------


def _summary(config: ExperimentConfig, view: FamilyView) -> dict[str, Any]:
    roles = view.evaluation["roles"]
    test = roles["test"]
    cls = test["classification"]
    focus = {
        label.value: {
            key: cls["per_class"][label.value][key]
            for key in ("precision", "recall", "f1")
        }
        for label in config.focus_labels
    }
    tuned = test["abstention"]["tuned_policy_calibrated"]
    reference = test["abstention"]["reference_policy_raw"]
    inference = view.runtime.get("inference", {})
    return {
        "selected_params": view.frozen["selected_params"],
        "test": {
            key: cls[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            )
        }
        | {
            "top2_accuracy": test["probabilities"]["raw"]["top2_accuracy"],
            "focus": focus,
            "log_loss_raw": test["probabilities"]["raw"]["log_loss"],
            "log_loss_calibrated": test["probabilities"]["calibrated"]["log_loss"],
            "brier_raw": test["probabilities"]["raw"]["brier"],
            "brier_calibrated": test["probabilities"]["calibrated"]["brier"],
            "ece_raw": test["probabilities"]["raw"]["ece"],
            "ece_calibrated": test["probabilities"]["calibrated"]["ece"],
            "aurc_calibrated": test["risk_coverage"]["calibrated"]["aurc"],
            "tuned_policy": {
                key: tuned[key]
                for key in (
                    "coverage",
                    "abstention_rate",
                    "selective_accuracy",
                    "selective_macro_f1",
                    "coverage_adjusted_accuracy",
                )
            },
            "reference_policy": {
                key: reference[key]
                for key in ("coverage", "abstention_rate", "selective_accuracy")
            },
        },
        "dev_tune_macro_f1": roles["dev_tune"]["classification"]["macro_f1"],
        "dev_calib_macro_f1": roles["dev_calib"]["classification"]["macro_f1"],
        "temperature": view.frozen["temperature"],
        "calibration_adopted": view.frozen["calibration_adopted"],
        "policy": view.frozen["policy"],
        "model_bytes": view.frozen["artifact_bytes"],
        "selected_fit_seconds": view.runtime.get("selected_fit_seconds"),
        "selected_vectorize_seconds": view.runtime.get("selected_vectorize_seconds"),
        "search_seconds": view.runtime.get("search_seconds"),
        "inference": inference,
    }


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _mb(value: int) -> str:
    return f"{value / 1e6:.1f} MB"


def markdown_report(
    config: ExperimentConfig,
    lock: dict[str, Any],
    comparison: dict[str, Any],
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    ud: dict[str, Any] | None,
    errors: list[dict[str, Any]],
) -> str:
    fams = list(comparison["models"])
    m = comparison["models"]
    lines: list[str] = []
    w = lines.append
    w(f"# Experiment {config.experiment_id}: fastText vs character TF-IDF + LR")
    w("")
    w(
        f"- Dataset: `{config.dataset.name}` {config.dataset.version}, train.jsonl "
        f"`{config.dataset.expected_sha256['train.jsonl'][:16]}…`"
    )
    w(
        f"- Selection lock: `{lock['lock_sha256'][:16]}…` (config `{config.sha256[:16]}…`)"
    )
    w(
        "- Protocol: parameters fit on **train**; hyperparameters chosen on **dev_tune** "
        "(macro-F1); temperature and abstention thresholds fit on **dev_calib**; "
        "**test** and **UD** scored only after the lock was written."
    )
    w(
        "- Scores are model estimates. Calibrated or not, they are not guaranteed "
        "probabilities of correctness."
    )
    w("")
    w("## Internal test (12,467 records, group-disjoint random split)")
    w("")
    header = "| Metric | " + " | ".join(DISPLAY[f] for f in fams) + " |"
    w(header)
    w("|---|" + "---:|" * len(fams))

    def row(name: str, values: list[str]) -> None:
        w(f"| {name} | " + " | ".join(values) + " |")

    t = {f: m[f]["test"] for f in fams}
    row("Accuracy", [_pct(t[f]["accuracy"]) for f in fams])
    row("Balanced accuracy", [_pct(t[f]["balanced_accuracy"]) for f in fams])
    row("**Macro F1** (primary)", [f"**{_num(t[f]['macro_f1'])}**" for f in fams])
    row("Weighted F1", [_num(t[f]["weighted_f1"]) for f in fams])
    row(
        "Macro precision / recall",
        [
            f"{_num(t[f]['macro_precision'])} / {_num(t[f]['macro_recall'])}"
            for f in fams
        ],
    )
    for label, name in (
        ("ckb", "Sorani"),
        ("kmr", "Kurmanji"),
        ("sdh", "Southern Kurdish"),
    ):
        row(f"{name} ({label}) F1", [_num(t[f]["focus"][label]["f1"]) for f in fams])
        row(
            f"{name} ({label}) recall",
            [_num(t[f]["focus"][label]["recall"]) for f in fams],
        )
    row("Top-2 accuracy", [_pct(t[f]["top2_accuracy"]) for f in fams])
    row("Model size on disk", [_mb(m[f]["model_bytes"]) for f in fams])
    row(
        "Training time, selected config",
        [
            f"{(m[f]['selected_fit_seconds'] or 0) + (m[f]['selected_vectorize_seconds'] or 0):.1f} s"
            for f in fams
        ],
    )
    row("Search time, whole grid", [f"{m[f]['search_seconds']:.0f} s" for f in fams])
    row(
        "Single-text latency, median / p95",
        [
            f"{m[f]['inference']['single_text_ms']['median']:.3f} / "
            f"{m[f]['inference']['single_text_ms']['p95']:.3f} ms"
            for f in fams
        ],
    )
    row(
        "Batch throughput",
        [
            f"{m[f]['inference']['batch']['texts_per_second']:,.0f} texts/s"
            for f in fams
        ],
    )
    row(
        "dev_tune macro F1 (selection, optimistic)",
        [_num(m[f]["dev_tune_macro_f1"]) for f in fams],
    )
    row(
        "dev_calib macro F1 (held out from selection)",
        [_num(m[f]["dev_calib_macro_f1"]) for f in fams],
    )
    w("")
    boot = comparison["bootstrap_test_macro_f1"]
    w(
        f"Paired cluster bootstrap over {boot['groups']:,} test document groups "
        f"({boot['resamples']} resamples, {int(boot['confidence'] * 100)}% intervals):"
    )
    w("")
    for family, values in boot["macro_f1"].items():
        lo, hi = values["interval"]
        w(f"- {DISPLAY[family]} macro-F1 {values['point']:.4f} [{lo:.4f}, {hi:.4f}]")
    for name, values in boot["differences"].items():
        lo, hi = values["interval"]
        w(
            f"- Difference {name.replace('tfidf_lr', 'TF-IDF').replace('fasttext', 'fastText')}: "
            f"{values['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]; left better in "
            f"{_pct(values['share_of_resamples_left_better'], 1)} of resamples"
        )
    w("")
    w("Selected configurations:")
    w("")
    for family in fams:
        w(
            f"- **{DISPLAY[family]}**: `{json.dumps(m[family]['selected_params'], sort_keys=True)}`"
        )
    w("")

    w("## Kurdish confusion (internal test)")
    w("")
    pairs = [f"{a.value}->{b.value}" for a, b in config.evaluation.errors.focus_pairs]
    w("| Confusion | " + " | ".join(DISPLAY[f] for f in fams) + " |")
    w("|---|" + "---:|" * len(fams))
    for pair in pairs:
        cells = []
        for f in fams:
            entry = analysis["models"][f]["focus_pairs"][pair]
            cells.append(f"{entry['count']} ({_pct(entry['rate_of_true_label'])})")
        w(f"| {pair} | " + " | ".join(cells) + " |")
    w("")
    for f in fams:
        rows = analysis["models"][f]["kurdish_rows"]
        w(f"{DISPLAY[f]}: where true Kurdish records went")
        w("")
        w("| true \\ predicted | " + " | ".join(config.label_values) + " |")
        w("|---|" + "---:|" * len(config.label_values))
        for label, entry in rows.items():
            w(
                f"| {label} (n={entry['support']}) | "
                + " | ".join(str(entry["predicted_as"][x]) for x in config.label_values)
                + " |"
            )
        w("")
    overlap = analysis.get("error_overlap")
    if overlap:
        w(
            f"Error overlap on Kurdish test records (n={overlap['records']}): "
            + ", ".join(
                f"{k.replace('_', ' ')} {v}"
                for k, v in overlap.items()
                if k not in {"scope", "records"}
            )
            + "."
        )
        w("")
    w("Conditions of the focus confusions (counts by length bucket / source / script):")
    w("")
    for f in fams:
        for pair in pairs:
            entry = analysis["models"][f]["focus_pairs"][pair]
            if entry["count"]:
                w(
                    f"- {DISPLAY[f]} `{pair}` ({entry['count']}): length {entry['by_length']}; "
                    f"source {entry['by_source']}; script {entry['by_script']}; mean "
                    f"confidence {_num(entry['mean_confidence'], 3)}; abstained "
                    f"{entry['abstained_under_tuned_policy']}"
                )
    w("")

    w("## Source, domain, length, script and contributor slices (internal test)")
    w("")
    w(
        f"Accuracy within each slice; slices with fewer than "
        f"{config.evaluation.min_slice_support} records are marked *low n*."
    )
    w("")
    for stem, title in (
        ("per_source", "Source × language"),
        ("per_domain", "Domain × language"),
        ("per_length", "Length (tokens) × language"),
        ("per_script", "Script × language"),
        ("per_contributor", "Contributor tier × language (tiers ranked on train)"),
        ("per_variety", "Variety × language"),
    ):
        w(f"### {title}")
        w("")
        w("| slice | language | n | " + " | ".join(DISPLAY[f] for f in fams) + " |")
        w("|---|---|---:|" + "---:|" * len(fams))
        index: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for r in tables[stem]:
            index[(r["slice"], r["language"])][r["model"]] = r
        for (slice_name, language), by_model in sorted(index.items()):
            if stem == "per_variety" and slice_name == "none" and language == "*":
                continue
            first = by_model[fams[0]]
            note = " *low n*" if first["low_support"] else ""
            w(
                f"| {slice_name} | {language} | {first['support']}{note} | "
                + " | ".join(_pct(by_model[f]["accuracy"]) for f in fams)
                + " |"
            )
        w("")

    w("## Calibration and abstention (internal test)")
    w("")
    w("| | " + " | ".join(DISPLAY[f] for f in fams) + " |")
    w("|---|" + "---:|" * len(fams))
    row(
        "Temperature (fit on dev_calib)",
        [
            f"{m[f]['temperature']:.4f} ({'adopted' if m[f]['calibration_adopted'] else 'not adopted'})"
            for f in fams
        ],
    )
    row(
        "Log loss raw → calibrated",
        [
            f"{t[f]['log_loss_raw']:.4f} → {t[f]['log_loss_calibrated']:.4f}"
            for f in fams
        ],
    )
    row(
        "Brier raw → calibrated",
        [f"{t[f]['brier_raw']:.4f} → {t[f]['brier_calibrated']:.4f}" for f in fams],
    )
    row(
        "ECE (15 bins) raw → calibrated",
        [f"{t[f]['ece_raw']:.4f} → {t[f]['ece_calibrated']:.4f}" for f in fams],
    )
    row("AURC (calibrated)", [_num(t[f]["aurc_calibrated"], 5) for f in fams])
    row(
        "Tuned policy (conf / margin)",
        [
            f"{m[f]['policy']['confidence_threshold']} / {m[f]['policy']['margin_threshold']}"
            for f in fams
        ],
    )
    row(
        "Tuned: coverage / selective accuracy",
        [
            f"{_pct(t[f]['tuned_policy']['coverage'])} / {_pct(t[f]['tuned_policy']['selective_accuracy'])}"
            for f in fams
        ],
    )
    row(
        "Reference 0.70/0.20 on raw: coverage / selective accuracy",
        [
            f"{_pct(t[f]['reference_policy']['coverage'])} / "
            f"{_pct(t[f]['reference_policy']['selective_accuracy'])}"
            for f in fams
        ],
    )
    w("")
    w(
        "Figures: `risk_coverage.png`, `reliability.png`, `confusion_fasttext.png`, "
        "`confusion_tfidf.png`."
    )
    w("")

    if ud:
        w("## UD v2.18 external benchmark")
        w("")
        w("### STRICT: Southern Kurdish (Garrusi), 217 records, Latin orthography")
        w("")
        w(
            "Every sdh training record is Arabic script; this slice combines an "
            "orthography, variety and domain shift and does not isolate script alone."
        )
        w("")
        _ud_table(w, ud["strict"], fams, config.label_values)
        w("### EXPLORATORY — prior repository exposure: Kurmanji")
        w("")
        w(
            "These UD Kurmanji records existed in the repository before this experiment. "
            "They are reported separately and never combined with the strict slice."
        )
        w("")
        _ud_table(w, ud["exploratory"], fams, config.label_values)

    w("## Representative errors")
    w("")
    w("Source labels are unreviewed; 'both agree' marks a candidate label error.")
    w("")
    shown = Counter()
    for record in errors:
        key = (record["model"], record["category"])
        if record["category"].startswith(("low_confidence", "other:")):
            continue
        if shown[key] >= config.evaluation.errors.markdown_examples_per_pair:
            continue
        shown[key] += 1
        text = (
            record["text"] if len(record["text"]) <= 120 else record["text"][:117] + "…"
        )
        w(
            f"- {DISPLAY[record['model']]} `{record['category']}` conf "
            f"{record['top1_confidence']:.3f}, {record['source']}, {record['tokens']} tok, "
            f"{record['script']}{', both agree' if record['both_models_agree_on_this_label'] else ''}: "
            f"{text}"
        )
    w("")
    w("## Limitations")
    w("")
    from kurdish_nlp.langid.experiments.runner import KNOWN_LIMITATIONS

    for item in KNOWN_LIMITATIONS:
        w(f"- {item}")
    w("")
    return "\n".join(lines)


def _ud_table(w, section: dict[str, Any], fams: list[str], labels) -> None:
    w("| | " + " | ".join(DISPLAY[f] for f in fams) + " |")
    w("|---|" + "---:|" * len(fams))
    s = {f: section[f] for f in fams}

    def r(name, values):
        w(f"| {name} | " + " | ".join(values) + " |")

    r(
        "Expected-label recall (argmax)",
        [
            f"{_pct(s[f]['argmax']['recall'])} ({s[f]['argmax']['correct']}/{s[f]['records']})"
            for f in fams
        ],
    )
    r(
        "Recall 95% Wilson interval",
        [
            f"[{_pct(s[f]['argmax']['recall_wilson_95'][0])}, {_pct(s[f]['argmax']['recall_wilson_95'][1])}]"
            for f in fams
        ],
    )
    r(
        "Predicted distribution",
        [
            ", ".join(
                f"{k} {v}"
                for k, v in s[f]["argmax"]["prediction_distribution"].items()
                if v
            )
            for f in fams
        ],
    )
    tuned = {f: s[f]["tuned_policy_calibrated"] for f in fams}
    r(
        "Tuned policy: abstention rate",
        [_pct(tuned[f]["abstention_rate"]) for f in fams],
    )
    r(
        "Tuned policy: selective recall",
        [_pct(tuned[f]["selective_recall"]) for f in fams],
    )
    r(
        "Tuned policy: coverage-adjusted recall",
        [_pct(tuned[f]["coverage_adjusted_recall"]) for f in fams],
    )
    r(
        "Tuned policy: distribution with und",
        [
            ", ".join(
                f"{k} {v}"
                for k, v in tuned[f]["prediction_distribution_with_und"].items()
                if v
            )
            for f in fams
        ],
    )
    r(
        "Mean calibrated score of expected label",
        [
            _num(s[f]["argmax"]["mean_calibrated_score_for_expected_label"], 3)
            for f in fams
        ],
    )
    w("")
    w("By length (argmax recall):")
    w("")
    for f in fams:
        parts = [
            f"{row['slice']} {row['correct']}/{row['support']}"
            for row in s[f]["by_length"]
            if row["language"] == "*"
        ]
        w(f"- {DISPLAY[f]}: " + "; ".join(parts))
    scripts = {
        row["slice"]: row["support"]
        for row in s[fams[0]]["by_script"]
        if row["language"] == "*"
    }
    w(f"- Scripts in slice: {scripts}")
    w("")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_reports(
    config: ExperimentConfig, lock: dict[str, Any], data: LoadedData, out_dir: Path
) -> dict[str, str]:
    paths = Paths(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = {family: FamilyView(config, paths, family, lock) for family in FAMILIES}
    labels = config.label_values
    test = data.test
    strict = None
    ud = None
    if all(view.external is not None for view in views.values()):
        texts = [
            e.text
            for s in (data.train, data.dev_tune, data.dev_calib, data.test)
            for e in s.examples
        ]
        external = load_external_benchmark(config, texts)
        strict = external.strict
        ud = {
            "benchmark": external.report,
            "lock_sha256": lock["lock_sha256"],
            "protocol": "models, temperatures and thresholds were frozen before this "
            "benchmark was scored; UD was never used for tuning, calibration or selection",
            "strict": {f: views[f].external["strict"] for f in FAMILIES},
            "exploratory": {f: views[f].external["exploratory"] for f in FAMILIES},
        }
    y = np.asarray(test.label_indices(config))
    predictions = {f: argmax(views[f].scores(test)[0]) for f in FAMILIES}
    boot = group_bootstrap(
        y,
        predictions,
        [e.group for e in test.examples],
        len(labels),
        resamples=config.evaluation.bootstrap_resamples,
        confidence=config.evaluation.bootstrap_confidence,
        seed=config.seed,
    )
    analysis, errors = kurdish_analysis(config, views, test, strict)
    tables = slice_tables(config, views, test)
    comparison = {
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "lock_sha256": lock["lock_sha256"],
        "primary_metric": "macro_f1 over the seven supervised classes (internal test)",
        "models": {f: _summary(config, views[f]) for f in FAMILIES},
        "bootstrap_test_macro_f1": boot,
        "development_roles": data.role_report,
        "contributors_train": data.contributor_report,
        "ud_external_summary": {
            key: {
                f: {
                    "recall": ud[key][f]["argmax"]["recall"],
                    "recall_wilson_95": ud[key][f]["argmax"]["recall_wilson_95"],
                    "prediction_distribution": ud[key][f]["argmax"][
                        "prediction_distribution"
                    ],
                    "tuned_abstention_rate": ud[key][f]["tuned_policy_calibrated"][
                        "abstention_rate"
                    ],
                }
                for f in FAMILIES
            }
            for key in ("strict", "exploratory")
        }
        if ud
        else None,
    }
    written: dict[str, str] = {}

    def text_file(name: str, content: str) -> None:
        path = out_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        written[name] = file_sha256(path)

    def json_file(name: str, value: Any) -> None:
        write_json(out_dir / name, value)
        written[name] = file_sha256(out_dir / name)

    json_file("comparison.json", comparison)
    json_file("kurdish_confusion.json", analysis)
    if ud:
        json_file("ud_external.json", ud)
    text_file("per_class.csv", _csv_text(per_class_rows(views), CLASS_COLUMNS))
    for stem, rows in sorted(tables.items()):
        text_file(f"{stem}.csv", _csv_text(rows, SLICE_COLUMNS))
    write_jsonl(out_dir / "errors.jsonl", errors)
    written["errors.jsonl"] = file_sha256(out_dir / "errors.jsonl")
    text_file(
        "comparison.md",
        markdown_report(config, lock, comparison, analysis, tables, ud, errors),
    )
    for family, name in (
        ("fasttext", "confusion_fasttext.png"),
        ("tfidf_lr", "confusion_tfidf.png"),
    ):
        matrix = views[family].evaluation["roles"]["test"]["classification"][
            "confusion_matrix"
        ]
        confusion_figure(out_dir / name, family, matrix["matrix"], labels)
        written[name] = file_sha256(out_dir / name)
    risk_coverage_figure(out_dir / "risk_coverage.png", views)
    written["risk_coverage.png"] = file_sha256(out_dir / "risk_coverage.png")
    reliability_figure(out_dir / "reliability.png", views)
    written["reliability.png"] = file_sha256(out_dir / "reliability.png")
    return written

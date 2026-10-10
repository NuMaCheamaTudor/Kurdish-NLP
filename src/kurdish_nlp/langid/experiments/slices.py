"""Metadata slices: language, source, domain, length, script, contributor.

Each slice row reports support, accuracy, macro-F1 over the labels present in
the slice, the most frequent wrong prediction and, when an abstention mask is
given, coverage and selective accuracy. Rows whose support is below the
configured minimum carry ``low_support = true`` and must not be read as
evidence on their own. For a single-language slice, accuracy is that
language's recall within the slice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from kurdish_nlp.langid.experiments.data import Example
from kurdish_nlp.langid.experiments.evaluation import macro_f1, rnd

ALL = "*"

DIMENSIONS: dict[str, Callable[[Example], str]] = {
    "language": lambda e: e.label.value,
    "source": lambda e: e.source_family,
    "domain": lambda e: e.domain,
    "length": lambda e: e.length_bucket,
    "script": lambda e: e.script,
    "contributor_tier": lambda e: e.contributor_tier,
    "variety": lambda e: e.variety,
    "review_flag": lambda e: "+".join(e.review_flags) or "none",
}

# (file stem, dimension, also split by language)
REPORT_TABLES: tuple[tuple[str, str, bool], ...] = (
    ("per_source", "source", True),
    ("per_domain", "domain", True),
    ("per_length", "length", True),
    ("per_script", "script", True),
    ("per_contributor", "contributor_tier", True),
    ("per_variety", "variety", True),
    ("per_review_flag", "review_flag", True),
)


def slice_rows(
    examples: Sequence[Example],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    dimension: str,
    *,
    by_language: bool,
    min_support: int,
    accepted: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    key = DIMENSIONS[dimension]
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, example in enumerate(examples):
        value = key(example)
        members[(value, ALL)].append(i)
        if by_language:
            members[(value, example.label.value)].append(i)
    rows = []
    for (value, language), indices in sorted(members.items()):
        if by_language and language == ALL and dimension == "language":
            continue
        idx = np.asarray(indices, dtype=np.int64)
        t, p = true[idx], pred[idx]
        correct = p == t
        wrong = Counter(labels[j] for j in p[~correct])
        top_wrong, top_wrong_count = (wrong.most_common(1) or [(None, 0)])[0]
        present = sorted({labels[j] for j in t})
        row: dict[str, Any] = {
            "dimension": dimension,
            "slice": value,
            "language": language,
            "support": len(idx),
            "correct": int(correct.sum()),
            "accuracy": rnd(correct.mean()),
            "macro_f1_present_labels": rnd(macro_f1(t, p, len(labels)))
            if len(present) > 1
            else None,
            "labels_present": "+".join(present),
            "top_wrong_prediction": top_wrong,
            "top_wrong_count": int(top_wrong_count),
            "low_support": bool(len(idx) < min_support),
        }
        if accepted is not None:
            kept = accepted[idx]
            covered = int(kept.sum())
            row["coverage"] = rnd(covered / len(idx))
            row["selective_accuracy"] = rnd(
                (correct & kept).sum() / covered if covered else None
            )
        rows.append(row)
    return rows


def predicted_distribution(
    y_pred: Sequence[int], labels: Sequence[str], accepted: np.ndarray | None = None
) -> dict[str, int]:
    """Counts of each predicted label; abstentions counted under ``und``."""

    pred = np.asarray(y_pred, dtype=np.int64)
    counts = {label: 0 for label in labels}
    for i, j in enumerate(pred):
        if accepted is not None and not accepted[i]:
            counts["und"] = counts.get("und", 0) + 1
        else:
            counts[labels[j]] += 1
    if accepted is not None:
        counts.setdefault("und", 0)
    return counts

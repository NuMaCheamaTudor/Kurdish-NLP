"""Exact entity metrics and explicit one-to-one span error diagnostics."""

from __future__ import annotations

from collections import Counter


def metrics(
    gold: list[set[tuple]], predicted: list[set[tuple]], labels: list[str]
) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("Reference/prediction length mismatch")
    tp, fp, fn, confusion, errors = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    examples = []
    for i, (gs, ps) in enumerate(zip(gold, predicted)):
        common = gs & ps
        tp.update(s[2] for s in common)
        fp.update(s[2] for s in ps - common)
        fn.update(s[2] for s in gs - common)
        remaining_g, remaining_p = set(gs - common), set(ps - common)
        candidates = []
        for g in remaining_g:
            for p in remaining_p:
                overlap = min(g[1], p[1]) - max(g[0], p[0])
                if overlap > 0:
                    exact_boundary = g[:2] == p[:2]
                    iou = overlap / (max(g[1], p[1]) - min(g[0], p[0]))
                    candidates.append((not exact_boundary, -iou, g, p))
        for _, _, g, p in sorted(candidates):
            if g not in remaining_g or p not in remaining_p:
                continue
            remaining_g.remove(g)
            remaining_p.remove(p)
            kind = (
                "type_only"
                if g[:2] == p[:2]
                else ("boundary_only" if g[2] == p[2] else "boundary_and_type")
            )
            errors[kind] += 1
            confusion[(g[2], p[2])] += 1
            if len(examples) < 100:
                examples.append({"index": i, "kind": kind, "gold": g, "predicted": p})
        errors["unmatched_gold"] += len(remaining_g)
        errors["unmatched_predictions"] += len(remaining_p)

    def score(t, p, n):
        support = t + n
        return {
            "precision": t / (t + p) if t + p else None,
            "recall": t / support if support else None,
            "f1": 2 * t / (2 * t + p + n) if support else None,
            "support": support,
            "predicted": t + p,
            "tp": t,
            "fp": p,
            "fn": n,
        }

    return {
        "micro": score(sum(tp.values()), sum(fp.values()), sum(fn.values())),
        "per_label": {
            label: score(tp[label], fp[label], fn[label])
            for label in sorted(set(labels) | set(tp) | set(fp) | set(fn))
        },
        "overlap_confusion": [
            {"gold": g, "predicted": p, "count": n}
            for (g, p), n in sorted(confusion.items())
        ],
        "span_errors": dict(errors),
        "error_examples": examples,
        "diagnostics_note": "Overlapping unmatched spans paired greedily by exact boundary then IoU, one to one. Remaining unmatched spans may overlap an already paired span (merge/split errors). Micro/per-label scores use strict (start,end,label) equality. None denotes undefined precision/unsupported gold label.",
    }

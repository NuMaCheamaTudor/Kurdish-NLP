"""Deterministic, exact threshold join over normalized token shingles.

Normalization is for leakage detection only. It never rewrites training text.
Prefix filtering considers every pair capable of reaching the Jaccard threshold;
this is not approximate MinHash or a capped nearest-neighbor search.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import Counter, defaultdict


def normalized_tokens(words: list[str]) -> tuple[str, ...]:
    fold = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ـ": "", "\u200c": ""})
    result = []
    for word in words:
        word = unicodedata.normalize("NFKC", word).translate(fold).casefold()
        word = "".join(c for c in word if unicodedata.category(c)[0] not in "PM")
        if word:
            result.append(word)
    return tuple(result)


def shingles(words: tuple[str, ...], width: int = 3) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(words[i : i + width]) for i in range(len(words) - width + 1))


def jaccard_pairs(sets: list[frozenset], threshold: float):
    """Yield all distinct nonempty-set pairs at or above threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0, 1]")
    frequency = Counter(s for values in sets for s in values)
    index: dict[tuple, list[int]] = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda i: (len(sets[i]), i)):
        values = sets[i]
        if not values:
            continue
        ordered = sorted(values, key=lambda s: (frequency[s], s))
        prefix = ordered[: len(values) - math.ceil(threshold * len(values)) + 1]
        candidates = {
            j
            for s in prefix
            for j in index[s]
            if len(sets[j]) + 1e-9 >= threshold * len(values)
        }
        for j in sorted(candidates):
            intersection = len(values & sets[j])
            score = intersection / (len(values) + len(sets[j]) - intersection)
            if score + 1e-12 >= threshold:
                yield j, i, score
        for s in prefix:
            index[s].append(i)


def group_records(records: list[dict], threshold: float = 0.85, min_tokens: int = 8):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    exact, normalized = {}, {}
    edges = []
    representatives, normalized_words = [], []
    for i, record in enumerate(records):
        # Malformed records may have no parsed tokens. Do not group empty text.
        words = tuple(record["tokens"])
        if not words:
            continue
        if words in exact:
            union(exact[words], i)
            edges.append((exact[words], i, "exact", 1.0))
        else:
            exact[words] = i
        norm = normalized_tokens(list(words))
        if not norm:
            continue
        if norm in normalized:
            union(normalized[norm], i)
            edges.append((normalized[norm], i, "normalized_exact", 1.0))
        else:
            normalized[norm] = i
            representatives.append(i)
            normalized_words.append(norm)
    sets = [
        shingles(words) if len(words) >= min_tokens else frozenset()
        for words in normalized_words
    ]
    for a, b, score in jaccard_pairs(sets, threshold):
        i, j = representatives[a], representatives[b]
        union(i, j)
        edges.append((i, j, "near", score))
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    groups = {}
    for members in components.values():
        # Content IDs avoid dependence on input ordering for group assignment.
        content = sorted({records[i]["text_sha256"] for i in members})
        gid = "g-" + hashlib.sha256("\n".join(content).encode()).hexdigest()
        for i in members:
            groups[records[i]["id"]] = gid
    return groups, [
        {
            "left": records[a]["id"],
            "right": records[b]["id"],
            "kind": kind,
            "similarity": score,
        }
        for a, b, kind, score in edges
    ]

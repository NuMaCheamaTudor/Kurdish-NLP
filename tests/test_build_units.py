"""Offline unit tests for the dataset-builder algorithms (no I/O)."""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations

import pytest

from kurdish_nlp.langid.acquisition.benchmark import ExternalBenchmarkRecord
from kurdish_nlp.langid.build.benchmark_guard import BenchmarkGuard
from kurdish_nlp.langid.build.config import MatchingConfig
from kurdish_nlp.langid.build.deduplicate import (
    ClusterMember,
    decide_cluster,
    similarity_join,
)
from kurdish_nlp.langid.build.grouping import UnionFind, mix_key, stable_key
from kurdish_nlp.langid.build.review import ReviewRecord
from kurdish_nlp.langid.build.sampling import (
    Slice,
    rate_matched_count,
    select_slices,
    water_fill,
)
from kurdish_nlp.langid.build.splitting import SplitGroup, assign_splits
from kurdish_nlp.langid.build.textkeys import (
    exact_key,
    loose_key,
    loose_text,
    script_class,
    sentence_loose_forms,
    shingles,
)

RATIOS = {"train": Fraction(4, 5), "dev": Fraction(1, 10), "test": Fraction(1, 10)}


# --------------------------------------------------------------- text keys
def test_loose_key_folds_case_punctuation_presentation_forms_and_joiners() -> None:
    assert loose_key("Hello, World!") == loose_key("hello world")
    assert loose_key("Ez çûm.") == loose_key("ez çûm")
    # Arabic presentation forms (U+FEDF U+FEE0 ...) fold by NFKC to base letters.
    assert loose_text("ﺍﻟﻌﺮﺑﻳﺔ") == "العربية"
    assert loose_key("می‌خواهم") == loose_key("میخواهم")
    assert loose_key("کـتاب") == loose_key("کتاب")  # tatweel
    assert loose_key("!!!") is None


def test_loose_key_never_folds_language_evidence_letters() -> None:
    # Arabic yeh/kaf versus Persian yeh/kaf must stay distinct.
    assert loose_key("كتابي") != loose_key("کتابی")
    # Sorani-specific letters are preserved.
    assert loose_key("ڕۆژ") != loose_key("روز")
    assert exact_key("Hello") != exact_key("hello")


def test_shingles_sentences_and_scripts() -> None:
    assert shingles("ab", 5) == frozenset({" ab "})
    assert " ez ç" in shingles(loose_text("Ez çûm"), 5)
    assert len(sentence_loose_forms("Yek du sê. Çar pênc şeş!")) == 2
    assert sentence_loose_forms("Only one sentence here") == ()
    assert script_class("Ez") == "latin"
    assert script_class("سڵاو") == "arabic"
    assert script_class("Ez سڵاو") == "mixed"
    assert script_class("123") == "none"


# --------------------------------------------------------------- grouping
def test_union_find_roots_are_order_independent_minimums() -> None:
    edges = [(5, 9), (9, 2), (7, 8), (2, 5), (1, 8)]
    roots = []
    for order in (edges, list(reversed(edges)), sorted(edges, key=lambda e: e[1])):
        uf = UnionFind(10)
        for left, right in order:
            uf.union(right, left)
        roots.append(list(uf.roots()))
    assert roots[0] == roots[1] == roots[2]
    assert roots[0][9] == 2 and roots[0][7] == 1 and roots[0][3] == 3


def test_stable_keys_are_deterministic_and_seeded() -> None:
    assert stable_key(42, "record", "a") == stable_key(42, "record", "a")
    assert stable_key(42, "record", "a") != stable_key(43, "record", "a")
    assert 0 <= mix_key(stable_key(1, "x"), 7) < 2**63


# ---------------------------------------------------------- deduplication
def _member(
    ord_: int, family: str, label: str, license_id: str = "MIT", eligible: bool = True
) -> ClusterMember:
    return ClusterMember(
        ord_, f"id-{ord_}", family, family, label, license_id, b"k", eligible
    )


def test_same_label_canonical_selection_is_deterministic_and_prioritized() -> None:
    members = [
        _member(1, "tatoeba", "sdh", "CC-BY-2.0-FR"),
        _member(7, "parme", "sdh"),
        _member(3, "wikimedia", "sdh", "CC-BY-SA-4.0"),
    ]
    priority = ("parme", "wikimedia", "tatoeba")
    licenses = ("CC0-1.0", "MIT", "CC-BY-2.0-FR", "CC-BY-SA-4.0")
    for permutation in (
        members,
        list(reversed(members)),
        [members[1], members[0], members[2]],
    ):
        decision = decide_cluster(
            permutation, source_priority=priority, license_priority=licenses
        )
        assert decision.kind == "same_label"
        assert decision.canonical == 7
        assert decision.duplicates == (1, 3)


def test_cross_label_cluster_is_ambiguous_and_ignores_excluded_members() -> None:
    priority, licenses = ("tatoeba",), ("MIT",)
    ambiguous = decide_cluster(
        [_member(1, "tatoeba", "en"), _member(2, "tatoeba", "tr")],
        source_priority=priority,
        license_priority=licenses,
    )
    assert ambiguous.kind == "cross_label" and ambiguous.ambiguous == (1, 2)
    single = decide_cluster(
        [_member(1, "tatoeba", "en"), _member(2, "tatoeba", "tr", eligible=False)],
        source_priority=priority,
        license_priority=licenses,
    )
    assert single.kind == "single_eligible" and single.canonical == 1


def _brute_force(
    items: list[tuple[int, frozenset[str]]], threshold: Fraction
) -> set[tuple[int, int]]:
    found = set()
    for (a, x), (b, y) in combinations(items, 2):
        if x and y and Fraction(len(x & y), len(x | y)) >= threshold:
            found.add((min(a, b), max(a, b)))
    return found


@pytest.mark.parametrize("threshold", [Fraction(1, 2), Fraction(4, 5), Fraction(1)])
def test_similarity_join_matches_brute_force(threshold: Fraction) -> None:
    rng = random.Random(7)
    words = ["ez", "tu", "diçim", "malê", "bajêr", "pirtûk", "nan", "av", "baş", "roj"]
    texts = []
    for _ in range(120):
        base = rng.sample(words, rng.randint(2, 6))
        if texts and rng.random() < 0.4:
            base = rng.choice(texts).split()
            if rng.random() < 0.7:
                base = base + [rng.choice(words)]
        texts.append(" ".join(base))
    items = [(index, shingles(loose_text(text), 5)) for index, text in enumerate(texts)]
    pairs, stats = similarity_join(items, threshold)
    assert {(pair.left, pair.right) for pair in pairs} == _brute_force(items, threshold)
    assert stats["candidate_pairs_verified"] <= len(items) * (len(items) - 1) // 2
    shuffled = items[:]
    rng.shuffle(shuffled)
    again, _ = similarity_join(shuffled, threshold)
    assert again == pairs


def test_similarity_join_keeps_distinct_dialect_examples_apart() -> None:
    # Two short Kurdish sentences that share a phrase are not near duplicates.
    a = shingles(loose_text("ئەمە کتێبی منە"), 5)
    b = shingles(loose_text("ئەمە کتاوەگەی منە"), 5)
    pairs, _ = similarity_join([(1, a), (2, b)], Fraction(4, 5))
    assert pairs == []


# ---------------------------------------------------------------- sampling
def test_water_fill_is_max_min_fair() -> None:
    assert water_fill(20, {"wiki": 3, "tatoeba": 100}) == {"tatoeba": 17, "wiki": 3}
    assert water_fill(9, {"a": 5, "b": 5, "c": 5}) == {"a": 3, "b": 3, "c": 3}
    assert water_fill(50, {"a": 5, "b": 10}) == {"a": 5, "b": 10}
    assert rate_matched_count(64, 18, 2000) == 0
    assert rate_matched_count(64_064, 18_904, 2_036_000) == 594


def _slices(spec: list[tuple[int, list[str | None]]]) -> list[Slice]:
    result, next_ord = [], 0
    for key, contributors in spec:
        members = tuple(range(next_ord, next_ord + len(contributors)))
        next_ord += len(contributors)
        result.append(Slice((key, members[0]), members, tuple(contributors)))
    return result


def test_select_slices_respects_quota_group_integrity_and_order() -> None:
    slices = _slices([(1, ["a", "b"]), (2, ["c", "c", "c"]), (3, ["d"]), (4, ["e"])])
    selection = select_slices(lambda: iter(slices), 4, contributor_cap=None)
    assert selection.selected == [0, 1, 5, 6]  # slice 2 (3 records) does not fit
    assert selection.skipped_too_large == 1
    for item in slices:
        chosen = set(item.members) & set(selection.selected)
        assert chosen in (set(), set(item.members))


def test_soft_contributor_cap_defers_then_fills() -> None:
    slices = _slices(
        [(1, ["big"]), (2, ["big"]), (3, ["big"]), (4, ["x"]), (5, ["big"])]
    )
    capped = select_slices(lambda: iter(slices), 3, contributor_cap=1)
    assert capped.contributor_counts["big"] == 2  # one pass-1 plus one pass-2 fill
    assert capped.selected == [0, 3, 1]
    assert capped.slices_pass2 == 1
    deterministic = select_slices(lambda: iter(slices), 3, contributor_cap=1)
    assert deterministic.selected == capped.selected


# --------------------------------------------------------------- splitting
def _groups(count: int, label: str = "en") -> list[SplitGroup]:
    return [
        SplitGroup(f"{label}-g{i}", (((label, "src", "-"), 1 + i % 3),))
        for i in range(count)
    ]


def test_assign_splits_is_deterministic_and_close_to_targets() -> None:
    groups = _groups(300) + _groups(120, "sdh")
    first, stats = assign_splits(groups, RATIOS, 42)
    second, _ = assign_splits(list(reversed(groups)), RATIOS, 42)
    assert first == second
    for stratum, by_split in stats["strata"].items():
        assert abs(by_split["train"]["realized_ratio"] - 0.8) < 0.02, stratum
        assert abs(by_split["test"]["realized_ratio"] - 0.1) < 0.02, stratum


def test_forced_giant_groups_go_to_train_and_multilabel_groups_stay_whole() -> None:
    giant = SplitGroup(
        "giant", ((("en", "t", "-"), 50), (("ckb", "t", "-"), 5)), "train"
    )
    multi = SplitGroup("multi", ((("en", "t", "-"), 1), (("tr", "t", "-"), 1)))
    assignment, stats = assign_splits([giant, multi, *_groups(40)], RATIOS, 1)
    assert assignment["giant"] == "train"
    assert assignment["multi"] in {"train", "dev", "test"}
    assert stats["forced_groups"] == 1
    with pytest.raises(ValueError, match="unique"):
        assign_splits([multi, multi], RATIOS, 1)


# --------------------------------------------------------- benchmark guard
def _benchmark(
    text: str, record_id: str = "ud:kmr:1"
) -> tuple[str, ExternalBenchmarkRecord]:
    return (
        "ud-kurdish-v2.18",
        ExternalBenchmarkRecord(
            record_id,
            text,
            "kmr",
            "ud-kurdish-v2.18",
            "kmr_kurmanji",
            "test",
            "external_test",
            "universal-dependencies",
            "CC-BY-SA-4.0",
            "slice",
            True,
            (),
        ),
    )


MATCHING = MatchingConfig(True, True, True, 5, Fraction(3, 5), Fraction(4, 5), 20, 3)


def _check(guard: BenchmarkGuard, text: str) -> list[str]:
    folded = loose_text(text)
    return [
        match.match_type
        for match in guard.check(
            exact=exact_key(text),
            loose=loose_key(text),
            folded=folded,
            sentence_forms=sentence_loose_forms(text),
        )
    ]


def test_benchmark_guard_match_layers() -> None:
    sentence = "Bavê min her roj bi trênê diçe kar û êvarê vedigere malê."
    guard = BenchmarkGuard([_benchmark(sentence)], MATCHING)
    assert _check(guard, sentence) == ["exact_normalized"]
    assert _check(
        guard, "bavê min her roj bi trênê diçe kar û êvarê vedigere malê!"
    ) == ["loose_normalized"]
    assert _check(guard, f"Ev destpêk e. {sentence} Ev dawî ye.") == [
        "contained_sentence"
    ]
    assert _check(guard, sentence[:-1] + " xwe.") == ["near_jaccard"]
    long = "Ji xwe re got ku " + sentence[:-1] + " bi lez û bez, piştî xebata dirêj"
    assert _check(guard, long) == ["near_containment"]
    assert _check(guard, "This sentence is unrelated to the benchmark.") == []


def test_benchmark_guard_ignores_short_sentence_fragments() -> None:
    # A benchmark record that starts with an interjection must not capture every
    # unrelated text that is just that interjection.
    guard = BenchmarkGuard(
        [_benchmark("Aha ! Rikab da bird rûwey gulabîyege .")], MATCHING
    )
    assert _check(guard, "Aha!") == []
    assert _check(guard, "Aha! Something else entirely here.") == []
    assert _check(guard, "aha! rikab da bird rûwey gulabîyege") == ["loose_normalized"]
    long_guard = BenchmarkGuard([_benchmark("Ez diçim malê. Tu jî were.")], MATCHING)
    assert _check(long_guard, "Ez diçim malê!") == ["contained_in_benchmark"]


# ------------------------------------------------------------------ review
def test_review_records_validate_decisions() -> None:
    base = {
        "schema_version": 1,
        "canonical_record_id": "tatoeba:en:1",
        "review_version": "1",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-11-01T10:00:00+00:00",
        "decision": "relabel",
        "corrected_label": "tr",
        "quality_status": "ok",
        "contamination_flags": [],
        "notes": None,
    }
    record = ReviewRecord.from_mapping(base)
    assert record.status == "reviewed:1" and record.excluded_reason() is None
    with pytest.raises(ValueError, match="trainable"):
        ReviewRecord.from_mapping({**base, "corrected_label": "und"})
    with pytest.raises(ValueError, match="only allowed"):
        ReviewRecord.from_mapping({**base, "decision": "accept"})
    rejected = ReviewRecord.from_mapping(
        {**base, "decision": "reject", "corrected_label": None}
    )
    assert rejected.excluded_reason() == "human_review_reject"

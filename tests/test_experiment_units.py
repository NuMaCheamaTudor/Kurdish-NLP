"""Offline unit tests for the language-ID experiment package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from experiment_world import LABELS, make_world

from kurdish_nlp.langid.build.splitting import SplitGroup
from kurdish_nlp.langid.dataset import (
    DatasetValidationError,
    format_fasttext_example,
    load_jsonl,
    write_fasttext_examples,
)
from kurdish_nlp.langid.experiments import calibration as cal
from kurdish_nlp.langid.experiments import evaluation as ev
from kurdish_nlp.langid.experiments.config import (
    ExperimentConfigError,
    load_experiment_config,
)
from kurdish_nlp.langid.experiments.data import (
    DatasetVerificationError,
    assign_group_roles,
    load_experiment_data,
    load_external_benchmark,
    pseudonymous_contributor,
)
from kurdish_nlp.langid.experiments.detector import (
    CalibratedLanguageDetector,
    load_experiment_detector,
)
from kurdish_nlp.langid.experiments.slices import slice_rows
from kurdish_nlp.langid.reliability import ReliabilityPolicy
from kurdish_nlp.langid.schemas import AbstentionReason, LanguageCode

REPO = Path(__file__).parents[1]
BASELINE = REPO / "configs/langid/experiments/baseline-v1.json"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_baseline_config_is_valid_and_bounded() -> None:
    config = load_experiment_config(BASELINE, repo_root=REPO)
    assert config.label_values == ("ckb", "kmr", "sdh", "ar", "fa", "tr", "en")
    assert "und" not in config.label_values
    assert config.selection_role == "dev_tune"
    assert config.calibration.fit_role == config.abstention.fit_role == "dev_calib"
    assert all(p["thread"] == 1 for p in config.fasttext.candidates())
    assert (
        len(config.fasttext.candidates()) <= config.resources.max_candidates["fasttext"]
    )


def _rewrite(path: Path, mutate) -> Path:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c.update(labels=[*c["labels"][:-1], "und"]), "und"),
        (lambda c: c["models"]["fasttext"]["fixed"].update(thread=4), "thread=1"),
        (
            lambda c: c["models"]["tfidf_lr"]["vectorizer_fixed"].update(
                strip_accents="unicode"
            ),
            "strip_accents",
        ),
        (lambda c: c["calibration"].update(fit_role="test"), "dev_calib"),
        (lambda c: c["selection"].update(role="test"), "dev_tune"),
        (lambda c: c["resources"]["max_candidates"].update(fasttext=1), "bound"),
        (
            lambda c: c["external_benchmark"]["exploratory"].update(title="Kurmanji"),
            "EXPLORATORY",
        ),
    ],
)
def test_config_rejects_protocol_violations(tmp_path: Path, mutation, message) -> None:
    path = _rewrite(make_world(tmp_path), mutation)
    with pytest.raises(ExperimentConfigError, match=message):
        load_experiment_config(path)


# --------------------------------------------------------------------------
# Dataset verification and roles
# --------------------------------------------------------------------------


def test_world_verifies_and_dev_roles_are_group_disjoint(tmp_path: Path) -> None:
    data = load_experiment_data(load_experiment_config(make_world(tmp_path)))
    tune = {e.group for e in data.dev_tune.examples}
    calib = {e.group for e in data.dev_calib.examples}
    assert tune and calib and not tune & calib
    assert {e.label.value for e in data.dev_tune.examples} == set(LABELS)
    assert {e.label.value for e in data.dev_calib.examples} == set(LABELS)
    assert len(data.dev_tune) + len(data.dev_calib) == 16 * 7
    assert set(data.calibration_folds) == {"fold_0", "fold_1"}
    train_groups = {e.group for e in data.train.examples}
    assert not train_groups & (tune | calib)
    assert data.verification["und_in_training_labels"] is False


def test_altered_dataset_file_fails_hash_verification(tmp_path: Path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    train = config.dataset.build_dir / "train.jsonl"
    train.write_bytes(train.read_bytes().replace(b'"label":"en"', b'"label":"tr"', 1))
    with pytest.raises(DatasetVerificationError, match="train.jsonl: sha256"):
        load_experiment_data(config)


def test_configured_hash_must_match_lock(tmp_path: Path) -> None:
    config_path = _rewrite(
        make_world(tmp_path),
        lambda c: c["dataset"]["expected_sha256"].update({"dev.jsonl": "0" * 64}),
    )
    with pytest.raises(DatasetVerificationError, match="differs from lock"):
        load_experiment_data(load_experiment_config(config_path))


def test_missing_dataset_fails_clearly(tmp_path: Path) -> None:
    config_path = _rewrite(
        make_world(tmp_path),
        lambda c: c["dataset"].update(build_dir=str(tmp_path / "missing")),
    )
    with pytest.raises(DatasetVerificationError, match="not rebuilt"):
        load_experiment_data(load_experiment_config(config_path))


def test_invalid_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = {
        "id": "x",
        "text": "valid text",
        "label": "und",
        "source": "s",
        "domain": "d",
        "document_id": "g",
        "license": "l",
        "split": "train",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="und"):
        load_jsonl(path)
    row["label"] = "de"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="unsupported label"):
        load_jsonl(path)


def test_assign_group_roles_is_deterministic_and_disjoint() -> None:
    groups = [
        SplitGroup(f"g{i}", ((("ckb" if i % 2 else "kmr", "src"), 1 + i % 3),))
        for i in range(200)
    ]
    first = assign_group_roles(groups, {"a": 0.5, "b": 0.5}, seed=1, salt="x")
    second = assign_group_roles(
        list(reversed(groups)), {"a": 0.5, "b": 0.5}, seed=1, salt="x"
    )
    assert first == second
    assert set(first) == {g.group_id for g in groups}
    sizes = {
        role: sum(g.size for g in groups if first[g.group_id] == role) for role in "ab"
    }
    assert abs(sizes["a"] - sizes["b"]) <= 3


def test_contributor_pseudonyms_hide_usernames() -> None:
    alias = pseudonymous_contributor("exp", "tatoeba", "SomeUser")
    assert alias.startswith("c-") and "SomeUser" not in alias
    assert alias == pseudonymous_contributor("exp", "tatoeba", "SomeUser")


# --------------------------------------------------------------------------
# fastText format
# --------------------------------------------------------------------------


def test_fasttext_format_is_one_line_per_record(tmp_path: Path) -> None:
    assert (
        format_fasttext_example("ckb", " دەق\nی کوردی ") == "__label__ckb دەق ی کوردی\n"
    )
    path = tmp_path / "out.txt"
    count = write_fasttext_examples(
        path, [("kmr", "rêz\r\nyek"), ("en", "line\u0085two"), ("sdh", "سێ")]
    )
    lines = path.read_text(encoding="utf-8").split("\n")
    assert count == 3 and lines[-1] == "" and len(lines) == 4
    assert lines[0] == "__label__kmr rêz yek"


@pytest.mark.parametrize(
    ("label", "text"), [("und", "text"), ("ckb", "   "), ("en", "a __label__fa b")]
)
def test_fasttext_format_rejects_unsafe_examples(label: str, text: str) -> None:
    with pytest.raises(ValueError):
        format_fasttext_example(label, text)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_confusion_matrix_rows_are_true_labels() -> None:
    matrix = ev.confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_classification_report_matches_scikit_learn() -> None:
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 7, 500))
    report = ev.classification_report(y, pred, LABELS)
    labels = list(range(7))
    assert report["macro_f1"] == pytest.approx(
        metrics.f1_score(y, pred, average="macro", labels=labels), abs=1e-6
    )
    assert report["weighted_f1"] == pytest.approx(
        metrics.f1_score(y, pred, average="weighted", labels=labels), abs=1e-6
    )
    assert report["balanced_accuracy"] == pytest.approx(
        metrics.balanced_accuracy_score(y, pred), abs=1e-6
    )
    per_class = metrics.precision_recall_fscore_support(y, pred, labels=labels)
    for i, label in enumerate(LABELS):
        assert report["per_class"][label]["precision"] == pytest.approx(
            per_class[0][i], abs=1e-6
        )
        assert report["per_class"][label]["recall"] == pytest.approx(
            per_class[1][i], abs=1e-6
        )
        assert report["per_class"][label]["support"] == per_class[3][i]
    assert (
        report["confusion_matrix"]["matrix"]
        == metrics.confusion_matrix(y, pred, labels=labels).tolist()
    )


def test_probability_metrics_known_values() -> None:
    proba = np.array([[0.8, 0.2], [0.4, 0.6]])
    y = np.array([0, 0])
    assert ev.log_loss(proba, y, 1e-12) == pytest.approx(
        -(np.log(0.8) + np.log(0.4)) / 2
    )
    assert ev.brier_score(proba, y) == pytest.approx((0.08 + 0.72) / 2)
    ece, table = ev.expected_calibration_error(proba, y, 10)
    # bin (0.7, 0.8] holds 0.8 (correct); bin (0.5, 0.6] holds 0.6 (wrong)
    assert ece == pytest.approx(0.5 * 0.2 + 0.5 * 0.6)
    assert sum(b["count"] for b in table) == 2


def test_risk_coverage_and_wilson() -> None:
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 0])
    curve = ev.risk_coverage(proba, y, points=5)
    assert curve["curve"][-1]["coverage"] == 1.0
    assert curve["curve"][-1]["risk"] == pytest.approx(0.25)
    assert curve["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4, abs=1e-6)
    low, high = ev.wilson_interval(0, 217, 0.95)
    assert low == 0.0 and high == pytest.approx(0.01739, abs=1e-4)


def test_vectorized_acceptance_equals_reliability_policy() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, size=(400, 7))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    texts = ["ab" if i % 17 == 0 else "valid words" for i in range(400)]
    letters = ev.letter_counts(texts)
    for confidence, margin in ((0.0, 0.0), (0.5, 0.1), (0.7, 0.2), (0.9, 0.5)):
        policy = ReliabilityPolicy(confidence, margin, 3)
        accepted = [
            lang != "und"
            for lang, _ in ev.policy_decisions(policy, texts, proba, LABELS)
        ]
        vectorized = ev.vectorized_acceptance(
            proba,
            letters,
            confidence_threshold=confidence,
            margin_threshold=margin,
            minimum_letter_count=3,
        )
        assert accepted == vectorized.tolist()


def test_group_bootstrap_is_deterministic_and_paired() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 120)
    good = y.copy()
    bad = np.where(rng.random(120) < 0.3, (y + 1) % 3, y)
    groups = [f"g{i // 2}" for i in range(120)]
    kwargs = {"resamples": 60, "confidence": 0.9, "seed": 5}
    a = ev.group_bootstrap(y, {"good": good, "bad": bad}, groups, 3, **kwargs)
    b = ev.group_bootstrap(y, {"good": good, "bad": bad}, groups, 3, **kwargs)
    assert a == b and a["groups"] == 60
    assert a["differences"]["bad-good"]["interval"][1] < 0


# --------------------------------------------------------------------------
# Calibration and thresholds
# --------------------------------------------------------------------------


def _overconfident(rng, n=3000, temperature=3.0):
    true_logits = rng.normal(0, 1.5, size=(n, 4))
    probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in probs])
    sharp = np.exp(true_logits * temperature)
    return sharp / sharp.sum(axis=1, keepdims=True), y


def test_temperature_scaling_recovers_overconfidence_and_keeps_argmax(tmp_path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    proba, y = _overconfident(np.random.default_rng(0))
    temperature = cal.fit_temperature(proba, y, config.calibration)
    assert temperature == pytest.approx(3.0, rel=0.15)
    scaled = cal.apply_temperature(proba, temperature, 1e-12)
    assert np.array_equal(scaled.argmax(axis=1), proba.argmax(axis=1))
    assert np.allclose(cal.apply_temperature(proba, 1.0, 1e-12), proba)
    folds = ["fold_0" if i % 2 else "fold_1" for i in range(len(y))]
    result = cal.calibrate(proba, y, folds, config.calibration)
    assert result["adopted"] is True
    assert (
        result["crossfit_temperature_scaling"]["log_loss"] < result["raw"]["log_loss"]
    )


def test_calibration_is_not_adopted_for_calibrated_scores(tmp_path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    proba, y = _overconfident(np.random.default_rng(2), temperature=1.0)
    folds = ["fold_0" if i % 2 else "fold_1" for i in range(len(y))]
    result = cal.calibrate(proba, y, folds, config.calibration)
    assert abs(result["temperature"] - 1.0) < 0.1


def test_threshold_selection_maximises_coverage_under_target(tmp_path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    proba = np.array([[0.95, 0.05]] * 90 + [[0.55, 0.45]] * 10)
    proba = np.hstack([proba, np.zeros((100, 5))])
    y = np.array([0] * 90 + [1] * 10)
    choice = cal.select_thresholds(proba, y, np.full(100, 10), config.abstention)
    assert choice["target_met"] and choice["dev_calib_coverage"] == 0.9
    assert choice["confidence_threshold"] == 0.9 or choice["margin_threshold"] == 0.2


# --------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------


def test_slices_report_support_and_low_support(tmp_path: Path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    data = load_experiment_data(config)
    examples = data.test.examples
    y = np.asarray(data.test.label_indices(config))
    rows = slice_rows(
        examples, y, y, config.label_values, "source", by_language=True, min_support=20
    )
    overall = [r for r in rows if r["language"] == "*"]
    assert sum(r["support"] for r in overall) == len(examples)
    assert all(r["accuracy"] == 1.0 for r in rows)
    assert any(r["low_support"] for r in rows)
    lengths = slice_rows(
        examples, y, y, config.label_values, "length", by_language=False, min_support=1
    )
    assert {r["slice"] for r in lengths} <= {"01-05", "06+"}


# --------------------------------------------------------------------------
# External benchmark isolation
# --------------------------------------------------------------------------


def test_external_benchmark_keeps_strict_and_exploratory_apart(tmp_path: Path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    external = load_external_benchmark(config)
    strict_ids = set(external.strict.ids)
    assert all(i.startswith("ud:sdh_garrusi") for i in strict_ids)
    assert not strict_ids & set(external.exploratory.ids)
    assert "ud:kmr_kurmanji:test:ov" not in external.exploratory.ids
    assert "ud:sdh_garrusi:test:dup" not in strict_ids
    assert external.report["exploratory"]["title"].startswith("EXPLORATORY")
    assert {e.label.value for e in external.exploratory.examples} == {"kmr"}


def test_external_benchmark_rejects_training_overlap(tmp_path: Path) -> None:
    config = load_experiment_config(make_world(tmp_path))
    with pytest.raises(DatasetVerificationError, match="overlap"):
        load_external_benchmark(config, ["Hewa emro serd bî."])


def test_external_benchmark_count_mismatch_fails(tmp_path: Path) -> None:
    path = _rewrite(
        make_world(tmp_path),
        lambda c: c["external_benchmark"]["strict"].update(expected_count=217),
    )
    with pytest.raises(DatasetVerificationError, match="strict"):
        load_external_benchmark(load_experiment_config(path))


# --------------------------------------------------------------------------
# Runtime detector
# --------------------------------------------------------------------------


class FixedClassifier:
    labels = LABELS

    def __init__(self, row) -> None:
        self.row = np.asarray(row, dtype=float)
        self.calls = 0

    def predict_proba(self, texts):
        self.calls += 1
        return np.tile(self.row, (len(texts), 1))


def _detector(row, **policy):
    classifier = FixedClassifier(row)
    detector = CalibratedLanguageDetector(
        classifier,
        temperature=1.0,
        policy=ReliabilityPolicy(
            **({"confidence_threshold": 0.5, "margin_threshold": 0.1} | policy)
        ),
        probability_floor=1e-12,
        model_identifier="test",
        model_version="0",
    )
    return detector, classifier


def test_detector_ranks_probabilities_and_abstains() -> None:
    detector, classifier = _detector([0.05, 0.02, 0.8, 0.03, 0.05, 0.03, 0.02])
    result = detector.detect("ئیمە دەقێک", top_k=2)
    assert result.language is LanguageCode.SDH and result.is_reliable
    assert [p.language.value for p in result.alternatives] == ["sdh", "ckb"]
    assert result.margin == pytest.approx(0.75)
    proba = detector.predict_proba("ئیمە دەقێک")
    assert list(proba)[:2] == ["sdh", "ckb"] and sum(proba.values()) == pytest.approx(1)
    empty = detector.detect("  \n ")
    assert empty.reason is AbstentionReason.EMPTY_TEXT and empty.alternatives == ()
    short = detector.detect("ab")
    assert short.reason is AbstentionReason.INSUFFICIENT_LEXICAL_CONTENT
    assert classifier.calls == 2
    with pytest.raises(ValueError, match="empty"):
        detector.predict_proba("")
    unsure, _ = _detector([0.4, 0.35, 0.05, 0.05, 0.05, 0.05, 0.05])
    assert unsure.detect("valid text").reason is AbstentionReason.LOW_CONFIDENCE
    batch = detector.detect_batch(["valid text", "", "دەقێکی کوردی"])
    assert [r.language.value for r in batch] == ["sdh", "und", "sdh"]


def test_missing_experiment_artifacts_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="inference spec"):
        load_experiment_detector(tmp_path)
    (tmp_path / "inference.json").write_text(
        json.dumps({"artifact": "model.bin", "artifact_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact not found"):
        load_experiment_detector(tmp_path)


# --------------------------------------------------------------------------
# fastText backend safety
# --------------------------------------------------------------------------


def test_fasttext_093_training_with_few_threads_is_refused() -> None:
    from kurdish_nlp.langid.experiments.fasttext_baseline import check_training_safety

    with pytest.raises(RuntimeError, match="uninitialized"):
        check_training_safety("0.9.3", 1)
    check_training_safety("0.9.3", 12)
    check_training_safety("0.9.2", 1)


def test_fasttext_probability_offset_is_removed_and_clipped() -> None:
    from kurdish_nlp.langid.fasttext_detector import fasttext_probability

    assert fasttext_probability(1.00001) == 1.0
    assert fasttext_probability(0.5 + 1e-5) == pytest.approx(0.5)
    assert fasttext_probability(0.0) == 0.0

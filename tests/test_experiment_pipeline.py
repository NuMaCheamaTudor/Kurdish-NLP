"""End-to-end experiment protocol on a tiny synthetic world.

Trains real (tiny) fastText and scikit-learn models in seconds. Real Dataset V1
training stays outside the pytest suite (``kurdish-langid-experiment all``).
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("fasttext")
pytest.importorskip("matplotlib")

from experiment_world import LABELS, make_world

from kurdish_nlp.langid import create_detector
from kurdish_nlp.langid.experiments import runner
from kurdish_nlp.langid.experiments.config import load_experiment_config
from kurdish_nlp.langid.experiments.data import (
    load_experiment_data,
    write_fasttext_training_file,
)
from kurdish_nlp.langid.experiments.detector import (
    load_experiment_detector,
)
from kurdish_nlp.langid.experiments.fasttext_baseline import FastTextModel
from kurdish_nlp.langid.experiments.tfidf_baseline import (
    TfidfLogisticModel,
)
from kurdish_nlp.langid.fasttext_detector import FastTextLanguageDetector
from kurdish_nlp.langid.schemas import AbstentionReason, LanguageCode

pytestmark = pytest.mark.experiment
QUIET = lambda message: None

SAMPLES = {
    "ckb": "ئەمڕۆ کەشوهەوا زۆر خۆشە",
    "kmr": "ez ê sibê biçim bajêr",
    "sdh": "هەر سوو چم ئەڕا مەدرەسە",
    "ar": "ذهب الولد إلى المدرسة صباحا",
    "fa": "امروز به کتابخانه رفتم",
    "tr": "bugün hava çok güzel",
    "en": "the teacher walked slowly to market",
}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("experiment")
    config = load_experiment_config(make_world(root))
    runner.run_preflight(config, QUIET)
    lock = runner.run_select(config, progress=QUIET)
    marker = runner.run_evaluate(config, QUIET)
    external = runner.run_external(config, QUIET)
    reports = runner.run_report(config, QUIET)
    reproduce = runner.run_reproduce(config, QUIET)
    return {
        "root": root,
        "config": config,
        "lock": lock,
        "marker": marker,
        "external": external,
        "reports": reports,
        "reproduce": reproduce,
    }


def test_pipeline_produces_every_artifact(world) -> None:
    root = world["config"].artifact_root
    for family in ("fasttext", "tfidf_lr"):
        for name in (
            "training.json",
            "calibration.json",
            "inference.json",
            "evaluation.json",
            "evaluation_external.json",
            "runtime.json",
        ):
            assert (root / family / name).is_file(), name
    assert (root / "fasttext/model.bin").is_file()
    assert (root / "tfidf_lr/model.joblib").is_file()
    assert (root / "tfidf_lr/model.joblib.meta.json").is_file()
    for name in (
        "comparison.json",
        "comparison.md",
        "per_class.csv",
        "per_source.csv",
        "per_domain.csv",
        "per_length.csv",
        "errors.jsonl",
        "ud_external.json",
        "confusion_fasttext.png",
        "confusion_tfidf.png",
        "risk_coverage.png",
    ):
        assert (root / "reports" / name).is_file(), name
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    assert manifest["model_artifacts_sha256"]
    assert manifest["dataset"]["files_sha256"]["train.jsonl"]


def test_internal_test_metrics_are_complete(world) -> None:
    evaluation = json.loads(
        (world["config"].artifact_root / "tfidf_lr/evaluation.json").read_text()
    )
    test = evaluation["roles"]["test"]
    assert test["records"] == 12 * 7
    cls = test["classification"]
    assert cls["confusion_matrix"]["labels"] == list(LABELS)
    assert sum(map(sum, cls["confusion_matrix"]["matrix"])) == 84
    assert set(cls["per_class"]) == set(LABELS)
    assert 0 <= cls["macro_f1"] <= 1
    assert test["abstention"]["none"]["coverage"] == 1.0
    tuned = test["abstention"]["tuned_policy_calibrated"]
    assert tuned["confusion_with_und"]["columns"][-1] == "und"
    assert (
        evaluation["reload_check_vs_selection"]["dev_tune"]["max_abs_difference"] == 0.0
    )


def test_selection_used_only_development_roles(world) -> None:
    lock = world["lock"]
    assert lock["protocol"]["test_used_for_selection"] is False
    assert lock["protocol"]["external_benchmark_used_for_selection"] is False
    calibration = json.loads(
        (world["config"].artifact_root / "fasttext/calibration.json").read_text()
    )
    assert calibration["calibration"]["fit_role"] == "dev_calib"
    assert calibration["abstention"]["fit_role"] == "dev_calib"
    training = json.loads(
        (world["config"].artifact_root / "fasttext/training.json").read_text()
    )
    assert training["selection"]["role"] == "dev_tune"
    assert training["search"]["fitted"] == 2


def test_reselection_after_test_is_refused(world) -> None:
    with pytest.raises(runner.ProtocolError, match="test-set tuning"):
        runner.run_select(world["config"], force=True, progress=QUIET)


def test_external_slices_are_separate_and_labelled(world) -> None:
    ud = json.loads(
        (world["config"].artifact_root / "reports/ud_external.json").read_text()
    )
    assert set(ud) >= {"strict", "exploratory", "benchmark", "protocol"}
    for family in ("fasttext", "tfidf_lr"):
        strict = ud["strict"][family]
        exploratory = ud["exploratory"][family]
        assert strict["status"] == "STRICT" and strict["records"] == 6
        assert exploratory["status"].startswith("EXPLORATORY")
        assert exploratory["records"] == 5
        assert "macro_f1" not in strict["argmax"]
        assert sum(strict["argmax"]["prediction_distribution"].values()) == 6


def test_reproducibility_is_verified(world) -> None:
    result = world["reproduce"]
    for family in ("fasttext", "tfidf_lr"):
        entry = result["families"][family]
        assert entry["reload"]["max_abs_difference"] == 0.0
        assert entry["retrain"]["argmax_identical"]
    assert result["families"]["fasttext"]["retrain"]["artifact_bitwise_identical"]
    regeneration = result["report_regeneration"]
    assert regeneration["different"] == [], regeneration


def test_detectors_load_and_cover_all_labels(world) -> None:
    root = world["config"].artifact_root
    for family in ("fasttext", "tfidf_lr"):
        detector = create_detector(backend="experiment", model_path=root / family)
        results = detector.detect_batch(list(SAMPLES.values()), top_k=7)
        for result in results:
            scores = [p.confidence for p in result.alternatives]
            assert scores == sorted(scores, reverse=True)
            assert {p.language for p in result.alternatives} == {
                LanguageCode(label) for label in LABELS
            }
        assert detector.detect("").reason is AbstentionReason.EMPTY_TEXT
        assert detector.detect(SAMPLES["kmr"]).to_dict()["model_version"] == "0.0.1"
        assert len(detector.predict_proba("ئەمڕۆ")) == 7
        raw = load_experiment_detector(root / family, calibrated=False)
        assert raw.temperature == 1.0


def test_tampered_artifact_is_refused(world, tmp_path) -> None:
    root = world["config"].artifact_root
    copy = tmp_path / "tfidf"
    copy.mkdir()
    for name in ("inference.json", "model.joblib", "model.joblib.meta.json"):
        (copy / name).write_bytes((root / "tfidf_lr" / name).read_bytes())
    with (copy / "model.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="sha256"):
        load_experiment_detector(copy)
    with pytest.raises(ValueError, match="refusing to unpickle"):
        TfidfLogisticModel.load(copy / "model.joblib")


def test_lock_detects_config_change_and_missing_lock(world, tmp_path) -> None:
    config_path = make_world(tmp_path)
    config = load_experiment_config(config_path)
    with pytest.raises(runner.ProtocolError, match="run 'select'"):
        runner.run_evaluate(config, QUIET)
    value = json.loads(world["config"].path.read_text())
    value["seed"] = 43
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runner.ProtocolError, match="configuration changed"):
        runner.verify_lock(load_experiment_config(changed))


def test_fasttext_model_alignment_save_and_reload(world, tmp_path) -> None:
    config = world["config"]
    data = load_experiment_data(config)
    train = write_fasttext_training_file(data.train, tmp_path / "train.txt")
    assert train["lines"] == len(data.train)
    params = config.fasttext.candidates()[-1]
    model = FastTextModel.train(train["path"], params, seed=config.seed, labels=LABELS)
    proba = model.predict_proba(list(SAMPLES.values()))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert [LABELS[i] for i in proba.argmax(axis=1)] == list(SAMPLES)
    model.save(tmp_path / "m.bin")
    reloaded = FastTextModel.load(tmp_path / "m.bin", LABELS)
    assert np.array_equal(reloaded.predict_proba(list(SAMPLES.values())), proba)
    again = FastTextModel.train(train["path"], params, seed=config.seed, labels=LABELS)
    again.save(tmp_path / "again.bin")
    assert (tmp_path / "m.bin").read_bytes() == (tmp_path / "again.bin").read_bytes()
    with pytest.raises(FileNotFoundError):
        FastTextModel.load(tmp_path / "missing.bin", LABELS)


def test_tfidf_columns_follow_configured_label_order(world, tmp_path) -> None:
    config = world["config"]
    data = load_experiment_data(config)
    reordered = tuple(reversed(LABELS))
    index = {label: i for i, label in enumerate(reordered)}
    y = [index[e.label.value] for e in data.train.examples]
    model = TfidfLogisticModel.fit(
        data.train.texts,
        y,
        vectorizer_fixed=config.tfidf.vectorizer_fixed,
        vectorizer_params=config.tfidf.vectorizer_candidates()[0],
        classifier_fixed=config.tfidf.classifier_fixed,
        classifier_params={"C": 10.0, "class_weight": None},
        labels=reordered,
        seed=1,
    )
    proba = model.predict_proba(list(SAMPLES.values()))
    assert [reordered[i] for i in proba.argmax(axis=1)] == list(SAMPLES)
    model.save(tmp_path / "m.joblib")
    reloaded = TfidfLogisticModel.load(tmp_path / "m.joblib", reordered)
    assert np.array_equal(reloaded.predict_proba(list(SAMPLES.values())), proba)
    with pytest.raises(ValueError, match="label order"):
        TfidfLogisticModel.load(tmp_path / "m.joblib", LABELS)


def test_existing_fasttext_adapter_works_with_real_model_on_numpy2(world) -> None:
    """Regression: fastText 0.9.3's single-text predict breaks under NumPy 2."""

    model_path = world["config"].artifact_root / "fasttext/model.bin"
    detector = FastTextLanguageDetector(
        model_path, confidence_threshold=0.0, margin_threshold=0.0
    )
    result = detector.detect(SAMPLES["tr"], top_k=7)
    assert all(0.0 <= p.confidence <= 1.0 for p in result.alternatives)
    assert sum(p.confidence for p in result.alternatives) == pytest.approx(
        1.0, abs=1e-4
    )

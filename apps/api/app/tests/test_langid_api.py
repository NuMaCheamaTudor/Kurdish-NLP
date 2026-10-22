"""Endpoints backed by the real baseline-v1 classifiers."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.languages import LANGUAGES
from app.tests.helpers import TRAIN_SAMPLES, requires_artifacts

pytestmark = requires_artifacts

EXPECTED_NAMES = {
    "ckb": ("Sorani", "Central Kurdish"),
    "kmr": ("Kurmanji", "Northern Kurdish"),
    "sdh": ("Southern Kurdish", None),
    "ar": ("Arabic", None),
    "fa": ("Persian", None),
    "tr": ("Turkish", None),
    "en": ("English", None),
}


def test_health_reports_ready(real_client: TestClient) -> None:
    response = real_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0", "language_id_ready": True}


def test_models_lists_both_classifiers_with_measured_metrics(real_client: TestClient) -> None:
    body = real_client.get("/api/v1/models").json()
    by_id = {model["id"]: model for model in body["models"]}

    tfidf, fasttext = by_id["tfidf_lr"], by_id["fasttext"]
    assert tfidf["status"] == fasttext["status"] == "ready"
    assert tfidf["default"] is True and fasttext["default"] is False
    assert body["default_model"] == "tfidf_lr"
    assert tfidf["languages"] == ["ckb", "kmr", "sdh", "ar", "fa", "tr", "en"]
    assert tfidf["identifier"] == "kurdish-langid-baseline-v1-tfidf_lr"
    # Values are read from the frozen evaluation files, not restated in code.
    assert tfidf["internal_test"] == {
        "scope": "internal",
        "dataset": "dataset-v1 1.0.0",
        "records": 12467,
        "macro_f1": 0.996516,
        "accuracy": 0.996952,
    }
    assert fasttext["internal_test"]["macro_f1"] == 0.99458
    assert {
        p["name"]: (p["confidence_threshold"], p["margin_threshold"]) for p in tfidf["policies"]
    } == {
        "tuned": (0.0, 0.0),
        "reference": (0.7, 0.2),
    }
    garrusi = tfidf["external_benchmarks"][0]
    assert (garrusi["records"], garrusi["correct"], garrusi["recall"]) == (217, 0, 0.0)


def test_models_marks_spacy_tasks_unavailable(real_client: TestClient) -> None:
    body = real_client.get("/api/v1/models").json()
    by_id = {model["id"]: model for model in body["models"]}
    for model_id in ("kurdish-ner-v1", "kurdish-pos-v1"):
        assert by_id[model_id]["status"] == "unavailable"
        assert by_id[model_id]["unavailable_reasons"]
        assert by_id[model_id]["internal_test"] is None
    tasks = {task["id"]: task for task in body["tasks"]}
    assert tasks["language"]["available"] is True
    assert tasks["entities"] == {
        "id": "entities",
        "name": "Named entities",
        "available": False,
        "message": "NER model is not installed.",
        "languages": [],
    }
    assert tasks["grammar"]["available"] is False
    assert body["limits"] == {"max_text_chars": 5000}
    assert [language["code"] for language in body["languages"]] == list(LANGUAGES)


@pytest.mark.parametrize("model", ["tfidf_lr", "fasttext"])
@pytest.mark.parametrize("code", list(TRAIN_SAMPLES))
def test_detect_maps_all_seven_labels(real_client: TestClient, model: str, code: str) -> None:
    response = real_client.post(
        "/api/v1/detect", json={"text": TRAIN_SAMPLES[code], "model": model}
    )
    assert response.status_code == 200
    body = response.json()
    name, alt_name = EXPECTED_NAMES[code]
    assert body["language"] == {"code": code, "name": name, "alt_name": alt_name}
    assert body["is_reliable"] is True and body["abstained"] is False
    assert body["reason"] is None and body["reason_message"] is None
    assert body["model"]["id"] == model
    assert body["scores"][0]["code"] == code
    assert body["confidence"] == body["scores"][0]["score"]


def test_detect_defaults_to_tfidf(real_client: TestClient) -> None:
    body = real_client.post("/api/v1/detect", json={"text": TRAIN_SAMPLES["ckb"]}).json()
    assert body["model"] == {
        "id": "tfidf_lr",
        "name": "TF-IDF + LR",
        "identifier": "kurdish-langid-baseline-v1-tfidf_lr",
        "version": "1.0.0",
    }
    assert body["policy"]["name"] == "tuned" and body["policy"]["calibrated"] is True


def test_class_scores_serialize_as_ranked_distribution(real_client: TestClient) -> None:
    response = real_client.post("/api/v1/detect", json={"text": TRAIN_SAMPLES["sdh"]})
    body = json.loads(response.text)
    scores = body["scores"]
    assert len(scores) == 7
    assert {item["code"] for item in scores} == set(EXPECTED_NAMES)
    values = [item["score"] for item in scores]
    assert values == sorted(values, reverse=True)
    assert all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in values)
    assert sum(values) == pytest.approx(1.0, abs=1e-6)
    assert {item["name"] for item in scores} == {name for name, _ in EXPECTED_NAMES.values()}
    assert body["score_semantics"].startswith("calibrated score estimates")


def test_top_k_limits_candidates(real_client: TestClient) -> None:
    body = real_client.post("/api/v1/detect", json={"text": TRAIN_SAMPLES["en"], "top_k": 2}).json()
    assert len(body["scores"]) == 2


def test_detect_matches_package_detector_directly(real_client: TestClient) -> None:
    from kurdish_nlp.langid import create_detector

    from app.config import DEFAULT_ARTIFACTS_DIR

    detector = create_detector(backend="experiment", model_path=DEFAULT_ARTIFACTS_DIR / "fasttext")
    text = TRAIN_SAMPLES["kmr"]
    expected = detector.detect(text, top_k=7)
    body = real_client.post("/api/v1/detect", json={"text": text, "model": "fasttext"}).json()
    assert body["confidence"] == expected.confidence
    assert body["margin"] == expected.margin
    assert [(s["code"], s["score"]) for s in body["scores"]] == [
        (p.language.value, p.confidence) for p in expected.alternatives
    ]


def test_reference_policy_equals_uncalibrated_loader(real_client: TestClient) -> None:
    from kurdish_nlp.langid.experiments.detector import load_experiment_detector

    from app.config import DEFAULT_ARTIFACTS_DIR

    raw = load_experiment_detector(DEFAULT_ARTIFACTS_DIR / "tfidf_lr", calibrated=False)
    for text in (TRAIN_SAMPLES["tr"], "Tehran Istanbul"):
        expected = raw.detect(text, top_k=7)
        body = real_client.post("/api/v1/detect", json={"text": text, "policy": "reference"}).json()
        assert body["language"]["code"] == expected.language.value
        assert body["reason"] == (expected.reason.value if expected.reason else None)
        assert [s["score"] for s in body["scores"]] == [p.confidence for p in expected.alternatives]
        assert body["policy"] == {
            "name": "reference",
            "calibrated": False,
            "confidence_threshold": 0.7,
            "margin_threshold": 0.2,
            "minimum_letter_count": 3,
        }


@pytest.mark.parametrize("model", ["tfidf_lr", "fasttext"])
def test_reference_policy_abstains_on_ambiguous_text(real_client: TestClient, model: str) -> None:
    body = real_client.post(
        "/api/v1/detect", json={"text": "Tehran Istanbul", "model": model, "policy": "reference"}
    ).json()
    assert body["language"] == {"code": "und", "name": "Inconclusive", "alt_name": None}
    assert body["abstained"] is True and body["is_reliable"] is False
    assert body["reason"] == "low_confidence"
    assert body["reason_message"]
    # The strongest candidate's score is kept on abstention, as in DetectionResult.
    assert body["confidence"] == body["scores"][0]["score"] > 0


@pytest.mark.parametrize("text", ["ab", "12345 !!", "\u200b\u200b"])
def test_insufficient_letters_abstain_without_scores(real_client: TestClient, text: str) -> None:
    body = real_client.post("/api/v1/detect", json={"text": text}).json()
    assert body["language"]["code"] == "und"
    assert body["reason"] in {"insufficient_lexical_content", "empty_text"}
    assert body["scores"] == [] and body["confidence"] == 0.0
    assert body["warnings"] == []


def test_input_stats_and_text_is_not_altered(real_client: TestClient) -> None:
    text = "  " + TRAIN_SAMPLES["fa"] + "\n"
    body = real_client.post("/api/v1/detect", json={"text": text}).json()
    assert body["input"]["characters"] == len(text)
    assert body["input"]["script"] == "arabic"
    assert body["input"]["words"] == len(text.split())
    assert body["elapsed_ms"] >= 0


def test_short_and_latin_kurmanji_warnings(real_client: TestClient) -> None:
    body = real_client.post("/api/v1/detect", json={"text": "Ez diçim malê"}).json()
    assert body["language"]["code"] == "kmr"
    assert {w["code"] for w in body["warnings"]} == {"latin_script_sdh", "short_text"}
    long_sorani = real_client.post("/api/v1/detect", json={"text": TRAIN_SAMPLES["ckb"]}).json()
    assert long_sorani["warnings"] == []

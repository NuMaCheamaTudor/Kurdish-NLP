"""Task routing through /api/v1/analyze."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.tests.helpers import TRAIN_SAMPLES, requires_artifacts

pytestmark = requires_artifacts


def test_language_task_auto(real_client: TestClient) -> None:
    response = real_client.post("/api/v1/analyze", json={"text": TRAIN_SAMPLES["kmr"]})
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "language"
    assert body["language"] == {
        "code": "kmr",
        "name": "Kurmanji",
        "alt_name": "Northern Kurdish",
        "mode": "auto",
    }
    assert body["detection"]["language"]["code"] == "kmr"
    assert len(body["detection"]["scores"]) == 7
    assert body["entities"] is None and body["grammar"] is None


def test_language_task_with_manual_language_still_reports_detection(
    real_client: TestClient,
) -> None:
    body = real_client.post(
        "/api/v1/analyze",
        json={"text": TRAIN_SAMPLES["kmr"], "language": "ckb", "model": "fasttext"},
    ).json()
    assert body["language"]["mode"] == "manual"
    assert body["language"]["code"] == "ckb"
    assert body["detection"]["language"]["code"] == "kmr"
    assert body["detection"]["model"]["id"] == "fasttext"


@pytest.mark.parametrize(
    ("task", "message"),
    [("entities", "NER model is not installed."), ("grammar", "Grammar model is not installed.")],
)
@pytest.mark.parametrize("language", ["auto", "ckb", "kmr"])
def test_unavailable_spacy_task(
    real_client: TestClient, task: str, message: str, language: str
) -> None:
    response = real_client.post(
        "/api/v1/analyze",
        json={"text": TRAIN_SAMPLES["ckb"], "task": task, "language": language},
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "task_unavailable", "message": message, "details": {"task": task}}
    }


def test_analyze_rejects_unknown_model_even_with_manual_language(real_client: TestClient) -> None:
    response = real_client.post(
        "/api/v1/analyze",
        json={"text": "abc def", "task": "entities", "language": "ckb", "model": "bert"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


def test_analyze_rejects_unsupported_language_setting(real_client: TestClient) -> None:
    response = real_client.post("/api/v1/analyze", json={"text": "abc", "language": "de"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_analyze_rejects_unknown_task(real_client: TestClient) -> None:
    response = real_client.post("/api/v1/analyze", json={"text": "abc", "task": "sentiment"})
    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "task"

"""Validation, limits, error shapes, missing artifacts and configuration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ARTIFACTS_DIR, Settings
from app.main import create_app
from app.tests.helpers import make_settings


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(create_app(make_settings(artifacts_dir=tmp_path, max_text_chars=50))) as c:
        yield c


def test_missing_artifacts_keep_the_app_up(empty_artifacts_client: TestClient) -> None:
    health = empty_artifacts_client.get("/health").json()
    assert health["status"] == "ok" and health["language_id_ready"] is False

    body = empty_artifacts_client.get("/api/v1/models").json()
    langid = [m for m in body["models"] if m["task"] == "language"]
    assert {m["id"] for m in langid} == {"tfidf_lr", "fasttext"}
    for model in langid:
        assert model["status"] == "unavailable"
        assert model["default"] is False
        assert model["unavailable_reasons"] == ["Model files are missing."]
        assert model["internal_test"] is None
    assert body["default_model"] is None
    assert body["tasks"][0] == {
        "id": "language",
        "name": "Language",
        "available": False,
        "message": "No language ID model is available.",
        "languages": [],
    }


def test_missing_artifacts_never_return_scores(empty_artifacts_client: TestClient) -> None:
    for model in (None, "fasttext"):
        response = empty_artifacts_client.post(
            "/api/v1/detect", json={"text": "Ez diçim malê", "model": model}
        )
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "model_unavailable"
        assert "scores" not in response.text


@pytest.mark.skipif(
    not (DEFAULT_ARTIFACTS_DIR / "tfidf_lr" / "inference.json").is_file(),
    reason="baseline-v1 artifacts are not present",
)
def test_inference_spec_must_match_selection_lock(tmp_path: Path) -> None:
    for family in ("tfidf_lr", "fasttext"):
        (tmp_path / family).mkdir()
        shutil.copy(DEFAULT_ARTIFACTS_DIR / family / "inference.json", tmp_path / family)
    lock = json.loads((DEFAULT_ARTIFACTS_DIR / "selection.lock.json").read_text())
    lock["families"]["tfidf_lr"]["inference_spec_sha256"] = "0" * 64
    (tmp_path / "selection.lock.json").write_text(json.dumps(lock))

    with TestClient(create_app(make_settings(artifacts_dir=tmp_path))) as c:
        models = {m["id"]: m for m in c.get("/api/v1/models").json()["models"]}
    assert models["tfidf_lr"]["unavailable_reasons"] == [
        "Inference spec does not match the selection lock."
    ]
    # The spec matches but the 180 MB binary is absent from the temp copy.
    assert models["fasttext"]["unavailable_reasons"] == ["Model files are missing."]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(client: TestClient, text: str) -> None:
    for path in ("/api/v1/detect", "/api/v1/analyze"):
        response = client.post(path, json={"text": text})
        assert response.status_code == 422
        assert response.json() == {
            "error": {"code": "empty_text", "message": "Enter some text to analyze."}
        }


def test_text_length_limit_counts_code_points(client: TestClient) -> None:
    assert client.post("/api/v1/detect", json={"text": "👋" * 50}).json()["error"]["code"] == (
        "model_unavailable"  # within the limit; fails later only because no model is loaded
    )
    response = client.post("/api/v1/detect", json={"text": "a" * 51})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "text_too_long"
    assert response.json()["error"]["details"] == {"max_text_chars": 50, "length": 51}


def test_invalid_model_id(client: TestClient) -> None:
    response = client.post("/api/v1/detect", json={"text": "hello world", "model": "../../etc"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({}, "text"),
        ({"text": 42}, "text"),
        ({"text": "abc", "top_k": 0}, "top_k"),
        ({"text": "abc", "policy": "strict"}, "policy"),
        ({"text": "abc", "model_path": "/tmp/x.joblib"}, "model_path"),
    ],
)
def test_validation_errors_do_not_echo_input(client: TestClient, payload: dict, field: str) -> None:
    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"][0]["field"] == field
    assert "/tmp/x.joblib" not in response.text


def test_malformed_json(client: TestClient) -> None:
    response = client.post(
        "/api/v1/detect", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_body_size_limit(tmp_path: Path) -> None:
    app = create_app(make_settings(artifacts_dir=tmp_path, max_body_bytes=1024))
    with TestClient(app) as c:
        response = c.post("/api/v1/detect", json={"text": "x" * 2000})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

        def chunks():
            yield b'{"text": "'
            yield b"x" * 2000
            yield b'"}'

        streamed = c.post(
            "/api/v1/detect", content=chunks(), headers={"content-type": "application/json"}
        )
        assert streamed.status_code == 413


def test_unknown_route_and_method(client: TestClient) -> None:
    assert client.get("/api/v1/nope").json() == {
        "error": {"code": "not_found", "message": "Not found."}
    }
    assert client.get("/api/v1/detect").status_code == 405


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/detect",
        headers={"origin": "http://localhost:5173", "access-control-request-method": "POST"},
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.get("/health", headers={"origin": "https://example.com"})
    assert "access-control-allow-origin" not in denied.headers


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNLP_MAX_TEXT_CHARS", "1200")
    monkeypatch.setenv("KNLP_CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("KNLP_NER_MODEL_PATH", "")
    monkeypatch.setenv("KNLP_ARTIFACTS_DIR", "artifacts/elsewhere")
    settings = Settings.from_env()
    assert settings.max_text_chars == 1200
    assert settings.max_body_bytes >= 1200 * 4
    assert settings.cors_origins == ("http://a.test", "http://b.test")
    assert settings.ner_model_path is None
    assert settings.artifacts_dir.parts[-2:] == ("artifacts", "elsewhere")
    assert settings.artifacts_dir.is_absolute()

    monkeypatch.setenv("KNLP_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="explicit origins"):
        Settings.from_env()

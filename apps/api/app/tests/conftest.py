from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.helpers import ARTIFACTS_PRESENT, make_settings


@pytest.fixture(scope="session")
def real_client() -> Iterator[TestClient]:
    if not ARTIFACTS_PRESENT:
        pytest.skip("baseline-v1 model artifacts are not present")
    app = create_app(
        make_settings(
            ner_model_path=Path("/nonexistent/ner"),
            grammar_model_path=Path("/nonexistent/grammar"),
        )
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def empty_artifacts_client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(artifacts_dir=tmp_path))
    with TestClient(app) as client:
        yield client

"""spaCy task availability and routing, using a duck-typed stand-in pipeline.

The stand-in only exercises the adapter and routing code paths; the app itself
never fabricates analysis.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import build_service, create_app
from app.services.spacy_models import SpacyTaskModel, TaskModelSpec, missing_files
from app.tests.helpers import make_settings

TEXT = "سڵاو 👋 لە هەولێر"  # includes an astral character before the entity


class FakeMorph:
    def __init__(self, features: dict[str, str]) -> None:
        self.features = features

    def to_dict(self) -> dict[str, str]:
        return self.features


class FakeDoc:
    def __init__(self, text: str) -> None:
        start = text.index("هەولێر")
        self.ents = [
            SimpleNamespace(text="هەولێر", label_="LOC", start_char=start, end_char=start + 6)
        ]
        self.tokens = []
        for i, word in enumerate(text.split(" ")):
            idx = text.index(word)
            self.tokens.append(
                SimpleNamespace(
                    i=i,
                    text=word,
                    idx=idx,
                    lemma_=word,
                    pos_="X",
                    tag_="",
                    dep_="dep",
                    morph=FakeMorph({"Case": "Nom"} if i == 0 else {}),
                )
            )
        for token in self.tokens:
            token.head = self.tokens[0]
        self.sents = [SimpleNamespace(start_char=0, end_char=len(text))]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.tokens)

    def has_annotation(self, name: str) -> bool:
        return name == "SENT_START"


class FakeNlp:
    def __init__(self, pipe_names: list[str]) -> None:
        self.pipe_names = pipe_names
        self.meta = {"version": "9.9.9"}

    def get_pipe(self, _: str) -> Any:
        return SimpleNamespace(labels=("LOC", "PER"))

    def __call__(self, text: str) -> FakeDoc:
        return FakeDoc(text)


def pipeline_dir(tmp_path: Path, pipeline: list[str], weights: bool = True) -> Path:
    path = tmp_path / "pipeline"
    path.mkdir()
    (path / "config.cfg").write_text("[nlp]\n", encoding="utf-8")
    (path / "meta.json").write_text(json.dumps({"pipeline": pipeline}), encoding="utf-8")
    if weights and "transformer" in pipeline:
        (path / "transformer").mkdir()
        (path / "transformer" / "model").write_bytes(b"weights")
    return path


def spec(path: Path | None, task: str = "entities") -> TaskModelSpec:
    return TaskModelSpec(
        id=f"test-{task}",
        task=task,  # type: ignore[arg-type]
        name="Test model",
        description="",
        languages=("ckb",),
        path=path,
        required_components=(("ner",),)
        if task == "entities"
        else (("tagger", "morphologizer"), ("parser",)),
    )


def test_missing_files_reasons(tmp_path: Path) -> None:
    assert missing_files(tmp_path / "absent") == ["Model directory not found."]
    path = pipeline_dir(tmp_path, ["transformer", "ner"], weights=False)
    assert missing_files(path) == ["Transformer weights are missing from this checkout."]


def test_task_model_states(tmp_path: Path) -> None:
    unconfigured = SpacyTaskModel(spec(None))
    unconfigured.load()
    assert not unconfigured.ready and unconfigured.reasons == ["No model is configured."]

    path = pipeline_dir(tmp_path, ["transformer", "ner"])
    failing = SpacyTaskModel(spec(path), loader=lambda _: (_ for _ in ()).throw(OSError("boom")))
    failing.load()
    assert not failing.ready and failing.reasons == ["The model failed to load."]

    wrong = SpacyTaskModel(spec(path), loader=lambda _: FakeNlp(["transformer", "tagger"]))
    wrong.load()
    assert not wrong.ready and "ner" in wrong.reasons[0]

    ready = SpacyTaskModel(spec(path), loader=lambda _: FakeNlp(["transformer", "ner"]))
    ready.load()
    assert ready.ready and ready.reasons == [] and ready.labels == ["LOC", "PER"]


def test_legacy_models_are_unavailable_in_this_checkout() -> None:
    # The repository's legacy folders lack transformer weights and spaCy is not
    # a dependency, so both tasks must report unavailable rather than guess.
    from app.config import DEFAULT_GRAMMAR_MODEL, DEFAULT_NER_MODEL

    for path, task in ((DEFAULT_NER_MODEL, "entities"), (DEFAULT_GRAMMAR_MODEL, "grammar")):
        model = SpacyTaskModel(spec(path, task))
        model.load()
        assert not model.ready and model.reasons


@pytest.fixture
def routed_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = make_settings(artifacts_dir=tmp_path / "no-artifacts")
    service = build_service(settings)
    path = pipeline_dir(tmp_path, ["transformer", "ner"])
    service.task_models = [
        SpacyTaskModel(spec(path), loader=lambda _: FakeNlp(["transformer", "ner"]))
    ]
    with TestClient(create_app(settings, service)) as client:
        yield client


def test_manual_language_routes_to_task_model(routed_client: TestClient) -> None:
    response = routed_client.post(
        "/api/v1/analyze", json={"text": TEXT, "task": "entities", "language": "ckb"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == {
        "code": "ckb",
        "name": "Sorani",
        "alt_name": "Central Kurdish",
        "mode": "manual",
    }
    assert body["detection"] is None
    entity = body["entities"]["entities"][0]
    # Code point offsets: slicing the submitted text recovers the entity.
    assert TEXT[entity["start"] : entity["end"]] == entity["text"] == "هەولێر"
    assert body["entities"]["labels"] == ["LOC", "PER"]
    assert body["entities"]["model"] == {
        "id": "test-entities",
        "name": "Test model",
        "version": "9.9.9",
    }


def test_manual_language_without_task_model_is_unsupported(routed_client: TestClient) -> None:
    response = routed_client.post(
        "/api/v1/analyze", json={"text": TEXT, "task": "entities", "language": "kmr"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "unsupported_language",
        "message": "Named entity recognition is not available for Kurmanji.",
        "details": {"task": "entities", "language": "kmr", "supported": ["ckb"]},
    }


def test_auto_routing_needs_a_language_id_model(routed_client: TestClient) -> None:
    response = routed_client.post("/api/v1/analyze", json={"text": TEXT, "task": "entities"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_grammar_adapter_output(tmp_path: Path) -> None:
    path = pipeline_dir(tmp_path, ["tagger", "parser"])
    model = SpacyTaskModel(spec(path, "grammar"), loader=lambda _: FakeNlp(["tagger", "parser"]))
    model.load()
    result = model.run(TEXT)
    assert [token.text for token in result.tokens] == TEXT.split(" ")
    for token in result.tokens:
        assert TEXT[token.start : token.end] == token.text
    assert result.tokens[0].morph == {"Case": "Nom"}
    assert result.tokens[2].head == 0
    assert [(s.start, s.end) for s in result.sentences] == [(0, len(TEXT))]

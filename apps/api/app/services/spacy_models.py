"""spaCy pipelines for named entities and grammar.

A task model is ``ready`` only after the pipeline has actually been loaded and
has the components its task needs. Missing files, a missing spaCy install, a
transformer component without its weights, or a load failure all leave the
model unavailable with a stated reason; nothing is ever simulated, and the
Language ID classifier is never used as a stand-in.

spaCy runs on the text exactly as submitted, so the character offsets it
reports are Unicode code point offsets into the user's text.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.schemas.analyze import (
    EntitiesResult,
    Entity,
    GrammarResult,
    Sentence,
    TaskModelRef,
    Token,
)

logger = logging.getLogger("kurdish_nlp.api.spacy")

TaskName = Literal["entities", "grammar"]


@dataclass(frozen=True, slots=True)
class TaskModelSpec:
    id: str
    task: TaskName
    name: str
    description: str
    languages: tuple[str, ...]
    path: Path | None
    # Any one of each inner tuple must be present in the loaded pipeline.
    required_components: tuple[tuple[str, ...], ...]


def spacy_installed() -> bool:
    return importlib.util.find_spec("spacy") is not None


def missing_files(path: Path) -> list[str]:
    """Pre-load checks that explain the common ways a pipeline is incomplete."""

    if not path.is_dir():
        return ["Model directory not found."]
    reasons = []
    if not (path / "config.cfg").is_file() or not (path / "meta.json").is_file():
        reasons.append("Pipeline config or meta.json is missing.")
        return reasons
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except ValueError:
        return ["meta.json is not valid JSON."]
    if "transformer" in meta.get("pipeline", []) and not (path / "transformer" / "model").is_file():
        reasons.append("Transformer weights are missing from this checkout.")
    return reasons


@dataclass
class SpacyTaskModel:
    spec: TaskModelSpec
    # Injectable for tests; defaults to ``spacy.load``.
    loader: Callable[[Path], Any] | None = None
    reasons: list[str] = field(default_factory=list)
    nlp: Any = None
    version: str | None = None
    labels: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.nlp is not None

    def load(self) -> None:
        """Load the pipeline if every prerequisite holds; otherwise record why not."""

        if self.nlp is not None:
            return
        loader = self.loader
        self.reasons = []
        if self.spec.path is None:
            self.reasons.append("No model is configured.")
            return
        self.reasons.extend(missing_files(self.spec.path))
        if loader is None and not spacy_installed():
            self.reasons.append("spaCy is not installed.")
        if self.reasons:
            return
        try:
            if loader is None:
                import spacy

                loader = spacy.load
            nlp = loader(self.spec.path)
        except Exception as error:  # noqa: BLE001 - reported as unavailable
            logger.warning("%s failed to load: %s", self.spec.id, type(error).__name__)
            self.reasons.append("The model failed to load.")
            return
        missing = [
            "/".join(options)
            for options in self.spec.required_components
            if not any(name in nlp.pipe_names for name in options)
        ]
        if missing:
            self.reasons.append(f"Pipeline lacks required components: {', '.join(missing)}.")
            return
        self.nlp = nlp
        self.version = str(getattr(nlp, "meta", {}).get("version") or "") or None
        if "ner" in nlp.pipe_names:
            self.labels = sorted(nlp.get_pipe("ner").labels)

    def ref(self) -> TaskModelRef:
        return TaskModelRef(id=self.spec.id, name=self.spec.name, version=self.version)

    def run(self, text: str) -> EntitiesResult | GrammarResult:
        assert self.nlp is not None
        with self.lock:
            doc = self.nlp(text)
        if self.spec.task == "entities":
            return entities_from_doc(doc, self.ref(), self.labels)
        return grammar_from_doc(doc, self.ref())


def entities_from_doc(doc: Any, model: TaskModelRef, labels: list[str]) -> EntitiesResult:
    entities = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]
    return EntitiesResult(
        model=model,
        labels=labels or sorted({entity.label for entity in entities}),
        entities=entities,
    )


def grammar_from_doc(doc: Any, model: TaskModelRef) -> GrammarResult:
    tokens = [
        Token(
            index=token.i,
            text=token.text,
            start=token.idx,
            end=token.idx + len(token.text),
            lemma=token.lemma_ or None,
            pos=token.pos_ or None,
            tag=token.tag_ or None,
            morph=dict(token.morph.to_dict()),
            dep=token.dep_ or None,
            head=token.head.i,
        )
        for token in doc
    ]
    sentences: list[Sentence] = []
    if getattr(doc, "has_annotation", lambda _: False)("SENT_START"):
        sentences = [Sentence(start=sent.start_char, end=sent.end_char) for sent in doc.sents]
    return GrammarResult(model=model, sentences=sentences, tokens=tokens)

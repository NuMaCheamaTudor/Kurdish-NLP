from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Language, LanguageSetting, PolicyName, StrictModel, Task
from app.schemas.detect import DetectResponse

OFFSETS = "Unicode code point offsets into the submitted text (end exclusive)."


class AnalyzeRequest(StrictModel):
    text: str
    task: Task = "language"
    language: LanguageSetting = Field(
        default="auto",
        description="'auto' routes by Language ID; a code overrides detection for routing.",
    )
    model: str | None = Field(default=None, description="Language ID model id.")
    policy: PolicyName = "tuned"


class TaskModelRef(BaseModel):
    id: str
    name: str
    version: str | None = None


class Entity(BaseModel):
    text: str
    label: str
    start: int = Field(description=OFFSETS)
    end: int = Field(description=OFFSETS)


class EntitiesResult(BaseModel):
    model: TaskModelRef
    labels: list[str]
    entities: list[Entity]


class Token(BaseModel):
    index: int
    text: str
    start: int = Field(description=OFFSETS)
    end: int = Field(description=OFFSETS)
    lemma: str | None
    pos: str | None
    tag: str | None
    morph: dict[str, str]
    dep: str | None
    head: int


class Sentence(BaseModel):
    start: int = Field(description=OFFSETS)
    end: int = Field(description=OFFSETS)


class GrammarResult(BaseModel):
    model: TaskModelRef
    sentences: list[Sentence]
    tokens: list[Token]


class LanguageResolution(Language):
    mode: Literal["auto", "manual"]


class AnalyzeResponse(BaseModel):
    task: Task
    language: LanguageResolution
    detection: DetectResponse | None = Field(
        description="Language ID output, present whenever the classifier ran."
    )
    entities: EntitiesResult | None = None
    grammar: GrammarResult | None = None

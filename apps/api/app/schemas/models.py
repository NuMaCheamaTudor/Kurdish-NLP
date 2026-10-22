from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import LanguageDetail, Task
from app.schemas.detect import PolicyInfo

Status = Literal["ready", "unavailable"]


class InternalTestMetrics(BaseModel):
    scope: Literal["internal"] = "internal"
    dataset: str
    records: int | None
    macro_f1: float
    accuracy: float | None


class ExternalBenchmark(BaseModel):
    name: str
    description: str
    records: int | None
    expected_label: str
    recall: float
    correct: int | None


class ModelInfo(BaseModel):
    id: str
    name: str
    full_name: str
    task: Task
    description: str
    languages: list[str]
    status: Status
    default: bool
    unavailable_reasons: list[str] = Field(default_factory=list)
    identifier: str | None = None
    version: str | None = None
    experiment: str | None = None
    framework: str | None = None
    size_bytes: int | None = None
    temperature: float | None = None
    policies: list[PolicyInfo] = Field(default_factory=list)
    artifact_sha256: str | None = None
    internal_test: InternalTestMetrics | None = None
    external_benchmarks: list[ExternalBenchmark] = Field(default_factory=list)


class TaskInfo(BaseModel):
    id: Task
    name: str
    available: bool
    message: str | None = Field(description="Why the task is unavailable, if it is.")
    languages: list[str]


class Limits(BaseModel):
    max_text_chars: int


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    tasks: list[TaskInfo]
    languages: list[LanguageDetail]
    limits: Limits
    default_model: str | None
    documentation_url: str

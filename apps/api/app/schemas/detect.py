from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Language, PolicyName, StrictModel


class DetectRequest(StrictModel):
    text: str = Field(description="Text to identify. Sent to the model unchanged.")
    model: str | None = Field(
        default=None, description="Language ID model id; the server default when omitted."
    )
    policy: PolicyName = Field(
        default="tuned",
        description=(
            "'tuned': calibrated scores with the frozen policy selected on dev_calib. "
            "'reference': raw scores with the 0.70 confidence / 0.20 margin reference "
            "policy recorded in the same inference spec."
        ),
    )
    top_k: int = Field(default=7, ge=1, le=7)


class ClassScore(BaseModel):
    code: str
    name: str
    score: float = Field(ge=0.0, le=1.0)


class ModelRef(BaseModel):
    id: str
    name: str
    identifier: str
    version: str


class PolicyInfo(BaseModel):
    name: PolicyName
    calibrated: bool
    confidence_threshold: float
    margin_threshold: float
    minimum_letter_count: int


class Notice(BaseModel):
    code: str
    severity: Literal["info", "warning"]
    message: str


class InputStats(BaseModel):
    characters: int
    letters: int
    words: int
    script: str


class DetectResponse(BaseModel):
    language: Language
    is_reliable: bool
    abstained: bool
    reason: str | None = Field(description="Abstention reason code when language is 'und'.")
    reason_message: str | None
    confidence: float = Field(
        description="Score of the top candidate (kept on abstention). A model estimate."
    )
    margin: float
    scores: list[ClassScore] = Field(description="Ranked candidates, highest first.")
    model: ModelRef
    policy: PolicyInfo
    score_semantics: str
    warnings: list[Notice]
    input: InputStats
    elapsed_ms: float

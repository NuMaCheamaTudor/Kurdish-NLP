from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LanguageSetting = Literal["auto", "ckb", "kmr", "sdh", "ar", "fa", "tr", "en"]
Task = Literal["language", "entities", "grammar"]
PolicyName = Literal["tuned", "reference"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Language(BaseModel):
    code: str = Field(description="Runtime language code; 'und' means inconclusive.")
    name: str
    alt_name: str | None = None


class LanguageDetail(Language):
    kurdish: bool
    script: str
    direction: Literal["rtl", "ltr", "auto"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail

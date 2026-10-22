from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.languages import LANGUAGES
from app.routes.deps import Service, run_inference
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.schemas.common import ErrorResponse, LanguageDetail
from app.schemas.detect import DetectRequest, DetectResponse
from app.schemas.models import Limits, ModelsResponse

DOCUMENTATION_URL = (
    "https://github.com/NuMaCheamaTudor/Kurdish-NLP/blob/main/docs/language-identification.md"
)

logger = logging.getLogger("kurdish_nlp.api")

router = APIRouter(
    prefix="/api/v1",
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)


@router.get("/models", response_model=ModelsResponse, tags=["models"])
async def list_models(request: Request, service: Service) -> ModelsResponse:
    await run_inference(request, service.langid.ensure_loaded)
    return ModelsResponse(
        models=service.langid.describe() + service.task_model_infos(),
        tasks=service.tasks(),
        languages=[
            LanguageDetail(
                code=info.code,
                name=info.name,
                alt_name=info.alt_name,
                kurdish=info.kurdish,
                script=info.script,
                direction=info.direction,  # type: ignore[arg-type]
            )
            for info in LANGUAGES.values()
        ],
        limits=Limits(max_text_chars=service.max_text_chars),
        default_model=service.langid.default_model,
        documentation_url=DOCUMENTATION_URL,
    )


@router.post("/detect", response_model=DetectResponse, tags=["language"])
async def detect(body: DetectRequest, request: Request, service: Service) -> DetectResponse:
    started = time.perf_counter()
    response = await run_inference(request, lambda: service.detect(body))
    # Metadata only: the submitted text is never logged.
    logger.info(
        "detect model=%s policy=%s chars=%d result=%s %.1fms",
        response.model.id,
        response.policy.name,
        response.input.characters,
        response.language.code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(body: AnalyzeRequest, request: Request, service: Service) -> AnalyzeResponse:
    started = time.perf_counter()
    response = await run_inference(request, lambda: service.analyze(body))
    logger.info(
        "analyze task=%s language=%s/%s chars=%d %.1fms",
        response.task,
        response.language.mode,
        response.language.code,
        len(body.text),
        (time.perf_counter() - started) * 1000,
    )
    return response

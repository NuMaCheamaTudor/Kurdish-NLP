"""FastAPI application factory.

Run locally with ``uvicorn app.main:app --app-dir apps/api --port 8000``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings
from app.errors import install_error_handlers
from app.middleware import BodySizeLimitMiddleware
from app.routes import health, v1
from app.services.analysis import AnalysisService, legacy_task_specs
from app.services.langid import LanguageIdService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_service(settings: Settings) -> AnalysisService:
    return AnalysisService(
        LanguageIdService(settings.artifacts_dir, settings.default_model),
        legacy_task_specs(settings.ner_model_path, settings.grammar_model_path),
        max_text_chars=settings.max_text_chars,
    )


def create_app(settings: Settings | None = None, service: AnalysisService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    analysis = service or build_service(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.inference_limiter = anyio.CapacityLimiter(settings.inference_concurrency)
        if settings.preload:
            await anyio.to_thread.run_sync(analysis.load)
        else:
            # Language ID loads on first use; spaCy availability is still checked
            # now so /models never reports an unchecked task as ready.
            for model in analysis.task_models:
                model.load()
        yield

    app = FastAPI(
        title="Kurdish NLP API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.analysis = analysis
    app.state.settings = settings
    install_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    app.include_router(health.router)
    app.include_router(v1.router)
    return app


app = create_app()

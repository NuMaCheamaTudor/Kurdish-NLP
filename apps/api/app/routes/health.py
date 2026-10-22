from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.routes.deps import Service

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    language_id_ready: bool


@router.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    return HealthResponse(
        status="ok", version=__version__, language_id_ready=service.langid.any_ready
    )

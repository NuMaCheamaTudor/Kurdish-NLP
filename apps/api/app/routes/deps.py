from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypeVar

import anyio
from fastapi import Depends, Request

from app.services.analysis import AnalysisService

T = TypeVar("T")


def get_service(request: Request) -> AnalysisService:
    return request.app.state.analysis


Service = Annotated[AnalysisService, Depends(get_service)]


async def run_inference(request: Request, function: Callable[[], T]) -> T:
    """Run blocking model code in a worker thread, bounded by the configured limit."""

    limiter: anyio.CapacityLimiter = request.app.state.inference_limiter
    return await anyio.to_thread.run_sync(function, limiter=limiter)

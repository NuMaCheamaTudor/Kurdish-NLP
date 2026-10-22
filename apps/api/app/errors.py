"""One error shape for every failure: ``{"error": {"code", "message", "details"}}``."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("kurdish_nlp.api")


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}


_HTTP_CODES = {
    404: ("not_found", "Not found."),
    405: ("method_not_allowed", "Method not allowed."),
    413: ("payload_too_large", "Request body is too large."),
}


def _validation_details(error: RequestValidationError) -> list[dict[str, Any]]:
    # Echo only the location and message, never the submitted value: it can be
    # the user's text.
    return [
        {
            "field": ".".join(str(part) for part in item.get("loc", ()) if part != "body"),
            "message": item.get("msg", "Invalid value."),
        }
        for item in error.errors()
    ]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            error_body(error.code, error.message, error.details),
            status_code=error.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body("invalid_request", "The request is invalid.", _validation_details(error)),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code, message = _HTTP_CODES.get(error.status_code, ("http_error", "Request failed."))
        return JSONResponse(
            error_body(code, message),
            status_code=error.status_code,
            headers=getattr(error, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            error_body("internal_error", "Something went wrong on the server."),
            status_code=500,
        )

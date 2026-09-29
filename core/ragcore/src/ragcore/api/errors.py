"""Errors the UI can translate: the English message plus a stable `code` and `params`.

The body stays `{"detail": <english>, ...}` so anything that only reads `detail`
keeps working; the desktop app maps `code` to its own catalog.
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, **params: object) -> None:
        super().__init__(status, message)
        self.code = code
        self.params = params


def api_error(status: int, code: str, message: str, **params: object) -> ApiError:
    return ApiError(status, code, message, **params)


async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.detail, "code": exc.code, "params": exc.params},
        status_code=exc.status_code,
        headers=exc.headers,
    )

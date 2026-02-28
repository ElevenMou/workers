"""FastAPI app assembly for Clipry workers API."""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from api_app.rate_limit import limiter
from api_app.routers.captions import router as captions_router
from api_app.routers.clips import router as clips_router
from api_app.routers.health import router as health_router
from api_app.routers.videos import router as videos_router
from api_app.routers.workers import router as workers_router
from config import validate_env


# ---------------------------------------------------------------------------
# Request body size limit middleware (default 1 MB for JSON endpoints)
# ---------------------------------------------------------------------------
_MAX_REQUEST_BODY_BYTES = int(os.getenv("API_MAX_REQUEST_BODY_BYTES", str(1_048_576)))


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_content_length: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                status_code=413,
                content={"error": "Request body too large"},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Request timeout middleware (default 60 s)
# ---------------------------------------------------------------------------
_REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "60"))


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "Request timed out"},
            )


app = FastAPI(title="Clipry Workers API", version="1.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."},
    )


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

# SECURITY: In production CORS_ALLOWED_ORIGINS must be set to the exact
# frontend URL. The localhost fallback is only for development.
cors_origin_regex = os.getenv("CORS_ALLOWED_ORIGIN_REGEX")
if not cors_allowed_origins and not cors_origin_regex:
    cors_allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RequestTimeoutMiddleware)
validate_env()

app.include_router(health_router)
app.include_router(workers_router)
app.include_router(captions_router)
app.include_router(videos_router)
app.include_router(clips_router)

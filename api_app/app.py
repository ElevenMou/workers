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
from api_app.routers.media import router as media_router
from api_app.routers.publishing import router as publishing_router
from api_app.routers.videos import router as videos_router
from api_app.routers.workers import router as workers_router
from config import validate_env
from utils.logging_config import correlation_id_var, generate_correlation_id, setup_logging
from utils.minio_client import initialize_minio_storage

setup_logging(component="api")


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


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Propagates or generates a correlation ID for every request."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        cid = request.headers.get("x-correlation-id") or generate_correlation_id()
        token = correlation_id_var.set(cid)
        try:
            response = await call_next(request)
            response.headers["x-correlation-id"] = cid
            return response
        finally:
            correlation_id_var.reset(token)


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

cors_origin_regex = os.getenv("CORS_ALLOWED_ORIGIN_REGEX")
if not cors_allowed_origins and not cors_origin_regex:
    _is_production = os.getenv("ENVIRONMENT", "").strip().lower() in {"production", "prod"}
    if _is_production:
        import logging as _cors_log
        _cors_log.getLogger(__name__).error(
            "CORS_ALLOWED_ORIGINS is not set in production. "
            "Requests from browser origins will be rejected. "
            "Set CORS_ALLOWED_ORIGINS to your frontend URL(s)."
        )
        cors_allowed_origins = []
    else:
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
app.add_middleware(CorrelationIdMiddleware)
validate_env(require_browser_cors=True)
initialize_minio_storage()

app.include_router(health_router)
app.include_router(workers_router)
app.include_router(captions_router)
app.include_router(videos_router)
app.include_router(clips_router)
app.include_router(publishing_router)
app.include_router(media_router)

# Versioned routes (v1 prefix) — mirrors root for forward-compatibility.
app.include_router(health_router, prefix="/v1")
app.include_router(workers_router, prefix="/v1")
app.include_router(captions_router, prefix="/v1")
app.include_router(videos_router, prefix="/v1")
app.include_router(clips_router, prefix="/v1")
app.include_router(publishing_router, prefix="/v1")
app.include_router(media_router, prefix="/v1")

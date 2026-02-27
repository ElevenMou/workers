"""FastAPI app assembly for Clipry workers API."""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api_app.rate_limit import limiter
from api_app.routers.captions import router as captions_router
from api_app.routers.clips import router as clips_router
from api_app.routers.health import router as health_router
from api_app.routers.videos import router as videos_router
from api_app.routers.workers import router as workers_router
from config import validate_env

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
validate_env()

app.include_router(health_router)
app.include_router(workers_router)
app.include_router(captions_router)
app.include_router(videos_router)
app.include_router(clips_router)

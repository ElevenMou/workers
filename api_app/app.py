"""FastAPI app assembly for Clipry workers API."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_app.routers.captions import router as captions_router
from api_app.routers.clips import router as clips_router
from api_app.routers.health import router as health_router
from api_app.routers.videos import router as videos_router
from api_app.routers.workers import router as workers_router
from config import validate_env

app = FastAPI(title="Clipry Workers API", version="1.0.0")
cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if not cors_allowed_origins:
    cors_allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_origin_regex=os.getenv(
        "CORS_ALLOWED_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    ),
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

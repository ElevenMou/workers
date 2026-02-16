"""FastAPI app assembly for Clipry workers API."""

from fastapi import FastAPI

from api_app.routers.captions import router as captions_router
from api_app.routers.clips import router as clips_router
from api_app.routers.health import router as health_router
from api_app.routers.videos import router as videos_router
from api_app.routers.workers import router as workers_router
from config import validate_env

app = FastAPI(title="Clipry Workers API", version="1.0.0")
validate_env()

app.include_router(health_router)
app.include_router(workers_router)
app.include_router(captions_router)
app.include_router(videos_router)
app.include_router(clips_router)

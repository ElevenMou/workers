"""Worker-scaling endpoints (admin only)."""

from fastapi import APIRouter, Depends, HTTPException, status

from api_app.auth import AuthenticatedUser, require_admin_user
from api_app.models import WorkerScaleRequest, WorkerScaleResponse
from api_app.state import logger
from config import NUM_CLIP_WORKERS, NUM_VIDEO_WORKERS
from utils.redis_client import (
    get_redis_connection,
    get_worker_scale_target,
    set_worker_scale_target,
)

router = APIRouter()


@router.get("/workers/scale", response_model=WorkerScaleResponse)
def get_worker_scale(
    _: AuthenticatedUser = Depends(require_admin_user),
) -> WorkerScaleResponse:
    """Return desired worker counts for the running worker supervisor."""
    try:
        conn = get_redis_connection()
        video_workers, clip_workers = get_worker_scale_target(
            connection=conn,
            default_video=NUM_VIDEO_WORKERS,
            default_clip=NUM_CLIP_WORKERS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read worker scale target: {exc}",
        ) from exc

    return WorkerScaleResponse(
        videoWorkers=video_workers,
        clipWorkers=clip_workers,
    )


@router.post("/workers/scale", response_model=WorkerScaleResponse)
def set_worker_scale(
    payload: WorkerScaleRequest,
    _: AuthenticatedUser = Depends(require_admin_user),
) -> WorkerScaleResponse:
    """Set desired worker counts. Worker supervisor applies changes at runtime."""
    if payload.videoWorkers is None and payload.clipWorkers is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of videoWorkers or clipWorkers",
        )

    try:
        conn = get_redis_connection()
        video_workers, clip_workers = set_worker_scale_target(
            connection=conn,
            video_workers=payload.videoWorkers,
            clip_workers=payload.clipWorkers,
            default_video=NUM_VIDEO_WORKERS,
            default_clip=NUM_CLIP_WORKERS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to update worker scale target: {exc}",
        ) from exc

    logger.info(
        "Worker scale target updated: video=%d clip=%d",
        video_workers,
        clip_workers,
    )
    return WorkerScaleResponse(
        videoWorkers=video_workers,
        clipWorkers=clip_workers,
    )

"""Clip generation endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from api_app.auth import AuthenticatedUser, get_current_user
from api_app.constants import CUSTOM_CLIP_TASK_PATH, GENERATE_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    CustomClipRequest,
    GenerateClipRequest,
    GenerateClipResponse,
)
from api_app.state import logger
from utils.supabase_client import supabase

router = APIRouter()


@router.post("/clips/generate", response_model=GenerateClipResponse)
def generate_clip(
    payload: GenerateClipRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GenerateClipResponse:
    """Enqueue generation for an existing suggested clip."""
    job_id = str(uuid4())
    user_id = current_user.id

    logger.info(
        "Generate clip request - user=%s  clip=%s",
        user_id,
        payload.clipId,
    )

    clip_resp = (
        supabase.table("clips")
        .select("id, video_id, user_id")
        .eq("id", payload.clipId)
        .execute()
    )
    raise_on_error(clip_resp, "Failed to load clip")

    clip_rows = clip_resp.data or []
    clip = clip_rows[0] if clip_rows else None
    if clip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip not found",
        )

    if clip["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clip does not belong to this user",
        )

    job_data = {
        "jobId": job_id,
        "clipId": payload.clipId,
        "userId": user_id,
        "layoutId": payload.layoutId,
    }

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": user_id,
                "video_id": clip["video_id"],
                "clip_id": payload.clipId,
                "type": "generate_clip",
                "status": "queued",
                "input_data": job_data,
            },
            on_conflict="id",
        )
        .execute()
    )
    raise_on_error(job_resp, "Failed to upsert job")

    enqueue_or_fail(
        queue_name="clip-generation",
        task_path=GENERATE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="generate_clip",
        video_id=clip["video_id"],
        clip_id=payload.clipId,
    )

    return GenerateClipResponse(
        jobId=job_id,
        clipId=payload.clipId,
        videoId=clip["video_id"],
        status="queued",
    )


@router.post("/clips/custom", response_model=GenerateClipResponse)
def custom_clip(
    payload: CustomClipRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GenerateClipResponse:
    """Create a clip from a URL + start/end time in one step."""
    if payload.endTime <= payload.startTime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="endTime must be greater than startTime",
        )
    duration = payload.endTime - payload.startTime
    if duration < 40 or duration > 90:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clip duration must be between 40 and 90 seconds",
        )

    clip_id = str(uuid4())
    job_id = str(uuid4())
    url_str = str(payload.url)
    user_id = current_user.id

    logger.info(
        "Custom clip request - user=%s  url=%s  %.2f-%.2f",
        user_id,
        url_str,
        payload.startTime,
        payload.endTime,
    )

    existing = (
        supabase.table("videos")
        .select("id")
        .eq("url", url_str)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    raise_on_error(existing, "Failed to query existing video")
    if existing.data and len(existing.data) > 0:
        video_id = existing.data[0]["id"]
        logger.info("Reusing existing video %s for url", video_id)
    else:
        video_id = str(uuid4())
        video_resp = (
            supabase.table("videos")
            .insert(
                {
                    "id": video_id,
                    "user_id": user_id,
                    "url": url_str,
                    "status": "pending",
                }
            )
            .execute()
        )
        raise_on_error(video_resp, "Failed to insert video")

    clip_insert = {
        "id": clip_id,
        "video_id": video_id,
        "user_id": user_id,
        "start_time": payload.startTime,
        "end_time": payload.endTime,
        "title": payload.title,
        "status": "pending",
    }
    if payload.layoutId:
        clip_insert["layout_id"] = payload.layoutId
    clip_resp = supabase.table("clips").insert(clip_insert).execute()
    raise_on_error(clip_resp, "Failed to insert clip")

    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "clipId": clip_id,
        "userId": user_id,
        "url": str(payload.url),
        "startTime": payload.startTime,
        "endTime": payload.endTime,
        "title": payload.title,
        "layoutId": payload.layoutId,
    }

    job_resp = (
        supabase.table("jobs")
        .insert(
            {
                "id": job_id,
                "user_id": user_id,
                "video_id": video_id,
                "clip_id": clip_id,
                "type": "generate_clip",
                "status": "queued",
                "input_data": job_data,
            }
        )
        .execute()
    )
    raise_on_error(job_resp, "Failed to insert job")

    enqueue_or_fail(
        queue_name="clip-generation",
        task_path=CUSTOM_CLIP_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="generate_clip",
        video_id=video_id,
        clip_id=clip_id,
    )

    return GenerateClipResponse(
        jobId=job_id,
        clipId=clip_id,
        videoId=video_id,
        status="queued",
    )

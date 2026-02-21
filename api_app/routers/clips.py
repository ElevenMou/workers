"""Clip generation endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

from api_app.access_rules import (
    enforce_clip_duration_limit,
    enforce_monthly_video_limit,
    enforce_processing_access_rules,
)
from api_app.auth import AuthenticatedUser, get_current_user
from api_app.constants import CUSTOM_CLIP_TASK_PATH, GENERATE_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    ClipLayoutOptionsResponse,
    CustomClipRequest,
    GenerateClipRequest,
    GenerateClipResponse,
)
from api_app.state import logger
from config import CREDIT_COST_CLIP_GENERATION
from utils.supabase_client import get_credit_balance, has_sufficient_credits, supabase

router = APIRouter()
_ACTIVE_GENERATE_JOB_STATUSES = ("queued", "processing", "retrying")


def _raise_if_insufficient_clip_generation_credits(*, user_id: str):
    required = int(CREDIT_COST_CLIP_GENERATION)
    if has_sufficient_credits(user_id=user_id, amount=required):
        return

    balance = get_credit_balance(user_id)
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=(
            "Insufficient credits for clip generation. "
            f"Required: {required}, available: {balance}."
        ),
    )


@router.get("/clips/layout-options", response_model=ClipLayoutOptionsResponse)
def clip_layout_options() -> ClipLayoutOptionsResponse:
    """Return supported clip canvas ratios and video scaling behavior."""
    from api_app.constants import (
        CANVAS_ASPECT_RATIO_OPTIONS,
        RECOMMENDED_CANVAS_ASPECT_RATIO,
        VIDEO_SCALE_MODE_OPTIONS,
    )

    return ClipLayoutOptionsResponse(
        aspectRatios=CANVAS_ASPECT_RATIO_OPTIONS,
        recommendedAspectRatio=RECOMMENDED_CANVAS_ASPECT_RATIO,
        videoScaleModes=VIDEO_SCALE_MODE_OPTIONS,
    )


@router.post("/clips/generate", response_model=GenerateClipResponse)
@limiter.limit("20/minute")
def generate_clip(
    request: Request,
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
        .select("id, video_id, user_id, start_time, end_time")
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

    clip_duration = float(clip["end_time"]) - float(clip["start_time"])
    enforce_clip_duration_limit(
        user_id=user_id,
        duration_seconds=clip_duration,
        supabase_client=supabase,
    )

    active_job_resp = (
        supabase.table("jobs")
        .select("id, status")
        .eq("user_id", user_id)
        .eq("clip_id", payload.clipId)
        .eq("type", "generate_clip")
        .in_("status", list(_ACTIVE_GENERATE_JOB_STATUSES))
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    raise_on_error(active_job_resp, "Failed to check existing active generation job")

    active_job_rows = active_job_resp.data or []
    if active_job_rows:
        active_job = active_job_rows[0]
        logger.info(
            "Reusing active clip generation job - user=%s clip=%s job=%s status=%s",
            user_id,
            payload.clipId,
            active_job.get("id"),
            active_job.get("status"),
        )
        return GenerateClipResponse(
            jobId=active_job["id"],
            clipId=payload.clipId,
            videoId=clip["video_id"],
            status=active_job["status"],
        )

    _raise_if_insufficient_clip_generation_credits(user_id=user_id)
    enforce_processing_access_rules(user_id, supabase_client=supabase)

    job_data = {
        "jobId": job_id,
        "clipId": payload.clipId,
        "userId": user_id,
        "layoutId": payload.layoutId,
        "generationCredits": int(CREDIT_COST_CLIP_GENERATION),
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
@limiter.limit("10/minute")
def custom_clip(
    request: Request,
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
    _raise_if_insufficient_clip_generation_credits(user_id=user_id)
    enforce_processing_access_rules(user_id, supabase_client=supabase)
    enforce_clip_duration_limit(
        user_id=user_id,
        duration_seconds=duration,
        supabase_client=supabase,
    )

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
        enforce_monthly_video_limit(user_id, supabase_client=supabase)
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
        "generationCredits": int(CREDIT_COST_CLIP_GENERATION),
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

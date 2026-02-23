"""Clip generation endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

from api_app.access_rules import (
    UserAccessContext,
    enforce_custom_clip_access,
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
from config import calculate_clip_generation_cost
from utils.supabase_client import (
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
)

router = APIRouter()
_ACTIVE_GENERATE_JOB_STATUSES = ("queued", "processing", "retrying")
_SMART_CLEANUP_ALLOWED_TIERS = {"pro", "enterprise"}
_STANDARD_CLIP_QUEUE = "clip-generation"
_PRIORITY_CLIP_QUEUE = "clip-generation-priority"


def _raise_if_insufficient_clip_generation_credits(
    *,
    context: UserAccessContext,
    actor_user_id: str,
    required_credits: int,
):
    required = int(required_credits)
    owner_user_id = context.billing_owner_user_id or actor_user_id
    if context.charge_source == "team_wallet" and context.workspace_team_id:
        has_credits = has_sufficient_credits(
            user_id=owner_user_id,
            amount=required,
            charge_source=context.charge_source,
            team_id=context.workspace_team_id,
        )
    else:
        has_credits = has_sufficient_credits(
            user_id=owner_user_id,
            amount=required,
        )
    if has_credits:
        return

    if context.charge_source == "team_wallet" and context.workspace_team_id:
        balance = get_team_wallet_balance(context.workspace_team_id)
        detail = (
            "Insufficient team credits for clip generation. "
            f"Required: {required}, team wallet available: {balance}."
        )
    else:
        balance = get_credit_balance(owner_user_id)
        detail = (
            "Insufficient credits for clip generation. "
            f"Required: {required}, available: {balance}."
        )

    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=detail,
    )


def _enforce_smart_cleanup_access(
    *,
    context: UserAccessContext,
    smart_cleanup_enabled: bool,
):
    if not smart_cleanup_enabled:
        return

    if context.tier in _SMART_CLEANUP_ALLOWED_TIERS:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Smart Cleanup is available for Pro and Enterprise plans only.",
    )


def _clip_queue_for_context(context: UserAccessContext) -> str:
    return _PRIORITY_CLIP_QUEUE if context.priority_processing else _STANDARD_CLIP_QUEUE


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
    smart_cleanup_enabled = bool(payload.smartCleanupEnabled)
    required_credits = calculate_clip_generation_cost(smart_cleanup_enabled)

    logger.info(
        "Generate clip request - user=%s clip=%s smart_cleanup=%s",
        user_id,
        payload.clipId,
        smart_cleanup_enabled,
    )
    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)

    clip_resp = (
        supabase.table("clips")
        .select("id, video_id, user_id, team_id, billing_owner_user_id, start_time, end_time")
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

    clip_team_id = clip.get("team_id")
    if access_context.workspace_team_id:
        if clip_team_id != access_context.workspace_team_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Clip does not belong to the active team workspace",
            )

        if access_context.workspace_role != "owner" and clip["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Team members can only generate clips they created.",
            )
    else:
        if clip_team_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Clip belongs to a team workspace. Switch to that workspace first.",
            )
        if clip["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Clip does not belong to this user",
            )

    active_job_query = (
        supabase.table("jobs")
        .select("id, status")
        .eq("clip_id", payload.clipId)
        .eq("type", "generate_clip")
        .in_("status", list(_ACTIVE_GENERATE_JOB_STATUSES))
    )
    if access_context.workspace_team_id:
        active_job_query = active_job_query.eq("team_id", access_context.workspace_team_id)
    else:
        active_job_query = active_job_query.eq("user_id", user_id).is_("team_id", "null")
    active_job_resp = (
        active_job_query
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

    clip_duration = float(clip["end_time"]) - float(clip["start_time"])
    enforce_clip_duration_limit(
        user_id=user_id,
        duration_seconds=clip_duration,
        supabase_client=supabase,
    )

    _enforce_smart_cleanup_access(
        context=access_context,
        smart_cleanup_enabled=smart_cleanup_enabled,
    )
    _raise_if_insufficient_clip_generation_credits(
        context=access_context,
        actor_user_id=user_id,
        required_credits=required_credits,
    )

    billing_owner_user_id = access_context.billing_owner_user_id or user_id
    job_data = {
        "jobId": job_id,
        "clipId": payload.clipId,
        "userId": user_id,
        "layoutId": payload.layoutId,
        "smartCleanupEnabled": smart_cleanup_enabled,
        "generationCredits": int(required_credits),
        "clipRetentionDays": access_context.clip_retention_days,
        "workspaceTeamId": access_context.workspace_team_id,
        "billingOwnerUserId": billing_owner_user_id,
        "chargeSource": access_context.charge_source,
        "workspaceRole": access_context.workspace_role,
    }

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": user_id,
                "team_id": access_context.workspace_team_id,
                "billing_owner_user_id": billing_owner_user_id,
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
        queue_name=_clip_queue_for_context(access_context),
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
    smart_cleanup_enabled = bool(payload.smartCleanupEnabled)
    required_credits = calculate_clip_generation_cost(smart_cleanup_enabled)

    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)
    enforce_custom_clip_access(context=access_context)
    _enforce_smart_cleanup_access(
        context=access_context,
        smart_cleanup_enabled=smart_cleanup_enabled,
    )
    _raise_if_insufficient_clip_generation_credits(
        context=access_context,
        actor_user_id=user_id,
        required_credits=required_credits,
    )
    enforce_clip_duration_limit(
        user_id=user_id,
        duration_seconds=duration,
        supabase_client=supabase,
    )

    logger.info(
        "Custom clip request - user=%s url=%s %.2f-%.2f smart_cleanup=%s",
        user_id,
        url_str,
        payload.startTime,
        payload.endTime,
        smart_cleanup_enabled,
    )

    existing_query = supabase.table("videos").select("id").eq("url", url_str)
    if access_context.workspace_team_id:
        existing_query = existing_query.eq("team_id", access_context.workspace_team_id)
    else:
        existing_query = existing_query.eq("user_id", user_id).is_("team_id", "null")
    existing = existing_query.limit(1).execute()
    raise_on_error(existing, "Failed to query existing video")
    billing_owner_user_id = access_context.billing_owner_user_id or user_id
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
                    "team_id": access_context.workspace_team_id,
                    "billing_owner_user_id": billing_owner_user_id,
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
        "team_id": access_context.workspace_team_id,
        "billing_owner_user_id": billing_owner_user_id,
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
        "smartCleanupEnabled": smart_cleanup_enabled,
        "generationCredits": int(required_credits),
        "clipRetentionDays": access_context.clip_retention_days,
        "workspaceTeamId": access_context.workspace_team_id,
        "billingOwnerUserId": billing_owner_user_id,
        "chargeSource": access_context.charge_source,
        "workspaceRole": access_context.workspace_role,
    }

    job_resp = (
        supabase.table("jobs")
        .insert(
            {
                "id": job_id,
                "user_id": user_id,
                "team_id": access_context.workspace_team_id,
                "billing_owner_user_id": billing_owner_user_id,
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
        queue_name=_clip_queue_for_context(access_context),
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

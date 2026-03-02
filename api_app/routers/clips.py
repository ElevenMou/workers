"""Clip generation endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from api_app.auth import get_user_rate_key
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
from api_app.rate_limit import limiter
from api_app.state import logger
from config import (
    calculate_clip_generation_cost,
    calculate_custom_clip_generation_cost,
)
from services.clips.quality_policy import resolve_clip_quality_policy
from utils.supabase_client import (
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
)

router = APIRouter()
_ACTIVE_GENERATE_JOB_STATUSES = ("queued", "processing", "retrying")
_SUCCESSFUL_GENERATE_JOB_STATUSES = ("completed",)
_SMART_CLEANUP_ALLOWED_TIERS = {"basic", "pro", "enterprise"}
_STANDARD_CLIP_QUEUE = "clip-generation"
_PRIORITY_CLIP_QUEUE = "clip-generation-priority"


def _raise_if_insufficient_clip_generation_credits(
    *,
    context: UserAccessContext,
    actor_user_id: str,
    required_credits: int,
):
    required = int(required_credits)
    if required <= 0:
        return

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


def _is_ai_suggested_clip(clip: dict) -> bool:
    if clip.get("ai_score") is not None:
        return True

    return clip.get("transcript_excerpt") is not None


def _clip_has_prior_generation(
    *,
    clip: dict,
    clip_id: str,
    access_context: UserAccessContext,
    actor_user_id: str,
) -> bool:
    clip_status = str(clip.get("status") or "").strip().lower()
    # A failed attempt should not consume the "first AI-suggested generation is free"
    # entitlement. Only successful generations count.
    if clip_status in _SUCCESSFUL_GENERATE_JOB_STATUSES:
        return True

    if clip.get("storage_path"):
        return True

    prior_job_query = (
        supabase.table("jobs")
        .select("id")
        .eq("clip_id", clip_id)
        .eq("type", "generate_clip")
        .in_("status", list(_SUCCESSFUL_GENERATE_JOB_STATUSES))
        .order("created_at", desc=True)
        .limit(1)
    )
    if access_context.workspace_team_id:
        prior_job_query = prior_job_query.eq("team_id", access_context.workspace_team_id)
    else:
        prior_job_query = prior_job_query.eq("user_id", actor_user_id).is_("team_id", "null")

    prior_job_resp = prior_job_query.execute()
    raise_on_error(prior_job_resp, "Failed to check prior generation jobs")
    return bool(prior_job_resp.data or [])


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
        detail="Smart Cleanup is available for Basic, Pro, and Enterprise plans only.",
    )


def _clip_queue_for_context(context: UserAccessContext) -> str:
    return _PRIORITY_CLIP_QUEUE if context.priority_processing else _STANDARD_CLIP_QUEUE


def _best_effort_delete_row(table: str, row_id: str):
    try:
        supabase.table(table).delete().eq("id", row_id).execute()
    except Exception as exc:
        logger.warning("Failed cleanup delete for %s %s: %s", table, row_id, exc)


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
@limiter.limit("15/minute", key_func=get_user_rate_key)
def generate_clip(
    request: Request,
    payload: GenerateClipRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GenerateClipResponse:
    """Enqueue generation for an existing suggested clip."""
    job_id = str(uuid4())
    user_id = current_user.id
    smart_cleanup_enabled = bool(payload.smartCleanupEnabled)

    logger.info(
        "Generate clip request - user=%s clip=%s smart_cleanup=%s",
        user_id,
        payload.clipId,
        smart_cleanup_enabled,
    )
    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)

    clip_resp = (
        supabase.table("clips")
        .select(
            "id, video_id, user_id, team_id, billing_owner_user_id, start_time, end_time, "
            "status, storage_path, ai_score, transcript_excerpt"
        )
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
    quality_policy = resolve_clip_quality_policy(
        tier=access_context.tier,
        clip_duration_seconds=clip_duration,
        requested_output_quality=None,
    )
    output_quality_override = (
        quality_policy.get("output_quality")
        if quality_policy.get("profile") == "premium_short_clip"
        else None
    )

    is_ai_suggested = _is_ai_suggested_clip(clip)
    has_prior_generation = _clip_has_prior_generation(
        clip=clip,
        clip_id=payload.clipId,
        access_context=access_context,
        actor_user_id=user_id,
    )
    required_credits = (
        0
        if is_ai_suggested and not has_prior_generation
        else calculate_clip_generation_cost(
            smart_cleanup_enabled,
            access_context.tier,
        )
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
    logger.info(
        "Clip generation credits resolved - user=%s clip=%s ai_suggested=%s prior_generation=%s required=%s",
        user_id,
        payload.clipId,
        is_ai_suggested,
        has_prior_generation,
        required_credits,
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
        "subscriptionTier": access_context.tier,
        "sourceMaxHeight": quality_policy.get("source_max_height"),
        "outputQualityOverride": output_quality_override,
        "qualityPolicyProfile": quality_policy.get("profile"),
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
@limiter.limit("5/minute", key_func=get_user_rate_key)
def custom_clip(
    request: Request,
    payload: CustomClipRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GenerateClipResponse:
    """Create a clip from a URL + start/end time in one step (always credit-consuming)."""
    if payload.endTime <= payload.startTime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="endTime must be greater than startTime",
        )
    duration = payload.endTime - payload.startTime
    if duration < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clip duration must be at least 10 seconds",
        )

    clip_id = str(uuid4())
    job_id = str(uuid4())
    url_str = str(payload.url)
    user_id = current_user.id
    smart_cleanup_enabled = bool(payload.smartCleanupEnabled)

    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)
    required_credits = calculate_custom_clip_generation_cost(
        smart_cleanup_enabled,
        access_context.tier,
    )
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
    created_video_for_request = False
    if existing.data and len(existing.data) > 0:
        video_id = existing.data[0]["id"]
        logger.info("Reusing existing video %s for url", video_id)
    else:
        enforce_monthly_video_limit(user_id, supabase_client=supabase)
        video_id = str(uuid4())
        created_video_for_request = True
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
    created_clip_for_request = True

    quality_policy = resolve_clip_quality_policy(
        tier=access_context.tier,
        clip_duration_seconds=duration,
        requested_output_quality=None,
    )
    output_quality_override = (
        quality_policy.get("output_quality")
        if quality_policy.get("profile") == "premium_short_clip"
        else None
    )

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
        "subscriptionTier": access_context.tier,
        "sourceMaxHeight": quality_policy.get("source_max_height"),
        "outputQualityOverride": output_quality_override,
        "qualityPolicyProfile": quality_policy.get("profile"),
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

    def _cleanup_custom_queue_full() -> None:
        if created_clip_for_request:
            _best_effort_delete_row("clips", clip_id)
        if created_video_for_request:
            _best_effort_delete_row("videos", video_id)

    enqueue_or_fail(
        queue_name=_clip_queue_for_context(access_context),
        task_path=CUSTOM_CLIP_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="generate_clip",
        video_id=video_id,
        clip_id=clip_id,
        on_queue_full_cleanup=_cleanup_custom_queue_full,
    )

    return GenerateClipResponse(
        jobId=job_id,
        clipId=clip_id,
        videoId=video_id,
        status="queued",
    )

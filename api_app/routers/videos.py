"""Video analysis and credit-cost endpoints."""

import asyncio
from math import ceil
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from api_app.auth import get_user_rate_key

from api_app.access_rules import (
    UserAccessContext,
    enforce_analysis_duration_limit,
    enforce_monthly_video_limit,
    enforce_processing_access_rules,
    get_user_access_context,
    is_analysis_duration_allowed,
)
from api_app.auth import AuthenticatedUser, get_current_user

from api_app.constants import ANALYZE_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    AnalyzeVideoRequest,
    AnalyzeVideoResponse,
    CreditsCostByUrlRequest,
    CreditsCostByUrlResponse,
)
from api_app.rate_limit import limiter
from api_app.state import logger, whisper_ready
from config import (
    calculate_video_analysis_cost,
)
from services.video_downloader import VideoDownloader
from utils.supabase_client import (
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
)

router = APIRouter()
_STANDARD_VIDEO_QUEUE = "video-processing"
_PRIORITY_VIDEO_QUEUE = "video-processing-priority"
_ANALYZE_CLIP_MIN_SECONDS = 10
_DEFAULT_ANALYZE_LEGACY_CLIP_SECONDS = 90


def _best_effort_delete_video(video_id: str):
    try:
        supabase.table("videos").delete().eq("id", video_id).execute()
    except Exception as exc:
        logger.warning("Failed cleanup delete for video %s: %s", video_id, exc)


def _video_queue_for_context(context: UserAccessContext) -> str:
    return _PRIORITY_VIDEO_QUEUE if context.priority_processing else _STANDARD_VIDEO_QUEUE


def _resolve_legacy_clip_range(*, clip_length_seconds: int, plan_max_seconds: int) -> tuple[int, int]:
    """
    Map legacy single-value clip selector to the historical analyze range buckets.
    """
    selected_max = max(
        _ANALYZE_CLIP_MIN_SECONDS,
        min(int(clip_length_seconds), int(plan_max_seconds)),
    )
    if selected_max <= 60:
        selected_min = _ANALYZE_CLIP_MIN_SECONDS
    elif selected_max <= 90:
        selected_min = 60
    elif selected_max <= 120:
        selected_min = 90
    else:
        selected_min = 120

    selected_min = min(selected_min, selected_max)
    selected_min = max(_ANALYZE_CLIP_MIN_SECONDS, selected_min)
    return selected_min, selected_max


def _normalize_processing_window(
    *,
    start_seconds: float,
    end_seconds: float | None,
    duration_seconds: int,
) -> tuple[float, float]:
    start = max(0.0, float(start_seconds))
    duration = max(0.0, float(duration_seconds))
    end = duration if end_seconds is None else min(duration, float(end_seconds))
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Processing end must be greater than processing start.",
        )
    return start, end


def _probe_credit_cost_for_url(url_str: str) -> dict:
    downloader = VideoDownloader()

    try:
        probe = downloader.probe_url(url_str)
    except Exception as exc:
        failure_reason = getattr(exc, "failure_reason", "probe_failed")
        logger.warning(
            "URL validation failed (%s) for %s: %s",
            failure_reason,
            url_str,
            exc,
        )
        return {
            "valid_url": False,
            "analysis_credits": 0,
            "duration_seconds": 0,
            "video_title": None,
            "thumbnail_url": None,
            "platform": None,
            "external_id": None,
            "has_captions": None,
            "has_audio": None,
            "detected_language": None,
        }

    duration_seconds = int(probe.get("duration_seconds") or 0)
    can_download = bool(probe.get("can_download"))
    has_source_captions = bool(probe.get("has_captions"))
    has_audio = bool(probe.get("has_audio"))
    can_use_whisper = False
    if not has_source_captions and has_audio:
        can_use_whisper = whisper_ready()
    has_captions = has_source_captions or can_use_whisper

    valid_url = can_download and has_captions and duration_seconds > 0
    if not valid_url:
        return {
            "valid_url": False,
            "analysis_credits": 0,
            "duration_seconds": duration_seconds,
            "video_title": probe.get("title"),
            "thumbnail_url": probe.get("thumbnail"),
            "platform": probe.get("platform"),
            "external_id": probe.get("external_id"),
            "has_captions": has_source_captions,
            "has_audio": has_audio,
            "detected_language": probe.get("detected_language"),
        }

    analysis_credits = calculate_video_analysis_cost(duration_seconds)
    return {
        "valid_url": True,
        "analysis_credits": analysis_credits,
        "duration_seconds": duration_seconds,
        "video_title": probe.get("title"),
        "thumbnail_url": probe.get("thumbnail"),
        "platform": probe.get("platform"),
        "external_id": probe.get("external_id"),
        "has_captions": has_source_captions,
        "has_audio": has_audio,
        "detected_language": probe.get("detected_language"),
    }


def _max_clip_count_for_duration(duration_seconds: int) -> int:
    """
    Enforce AI analyze clip density:
    maximum 1 requested clip per 3 minutes of source duration.
    """
    seconds = max(int(duration_seconds), 0)
    return max(1, seconds // 180)


def _raise_if_insufficient_credits(
    *,
    context: UserAccessContext,
    actor_user_id: str,
    required_credits: int,
):
    if required_credits <= 0:
        return

    owner_user_id = context.billing_owner_user_id or actor_user_id
    if context.charge_source == "team_wallet" and context.workspace_team_id:
        has_credits = has_sufficient_credits(
            user_id=owner_user_id,
            amount=required_credits,
            charge_source=context.charge_source,
            team_id=context.workspace_team_id,
        )
    else:
        has_credits = has_sufficient_credits(
            user_id=owner_user_id,
            amount=required_credits,
        )
    if has_credits:
        return

    if context.charge_source == "team_wallet" and context.workspace_team_id:
        balance = get_team_wallet_balance(context.workspace_team_id)
        detail = (
            "Insufficient team credits for analysis. "
            f"Required: {required_credits}, team wallet available: {balance}."
        )
    else:
        balance = get_credit_balance(owner_user_id)
        detail = (
            "Insufficient credits for analysis. "
            f"Required: {required_credits}, available: {balance}."
        )

    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=detail,
    )


@router.post("/videos/analyze", response_model=AnalyzeVideoResponse)
@limiter.limit("10/minute")
@limiter.limit("5/minute", key_func=get_user_rate_key)
async def analyze_video(
    request: Request,
    payload: AnalyzeVideoRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AnalyzeVideoResponse:
    """Create and enqueue a video analysis job."""
    job_id = str(uuid4())
    url_str = str(payload.url)
    user_id = current_user.id
    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)

    # Fast-fail if user has no credits at all (before URL probing).
    _raise_if_insufficient_credits(
        context=access_context,
        actor_user_id=user_id,
        required_credits=1,
    )

    # Enforce validation gate equivalent to /credits/cost before job enqueue.
    # Run in a thread with timeout to avoid blocking the event loop.
    try:
        url_probe = await asyncio.wait_for(
            asyncio.to_thread(_probe_credit_cost_for_url, url_str),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Video URL probe timed out. Please try again.",
        )
    if not url_probe["valid_url"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Video URL cannot be analyzed. Ensure it is downloadable and has "
                "captions (or audio for transcription)."
            ),
        )

    analysis_duration_seconds = int(url_probe.get("duration_seconds") or 0)
    enforce_analysis_duration_limit(
        context=access_context,
        duration_seconds=analysis_duration_seconds,
    )
    clip_length_plan_max = int(access_context.max_clip_duration_seconds)
    selected_clip_min: int
    selected_clip_max: int
    clip_length_legacy_value = payload.clipLengthSeconds
    clip_length_min_payload = payload.clipLengthMinSeconds
    clip_length_max_payload = payload.clipLengthMaxSeconds
    if (clip_length_min_payload is None) != (clip_length_max_payload is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "clipLengthMinSeconds and clipLengthMaxSeconds must be provided together."
            ),
        )

    if clip_length_min_payload is not None and clip_length_max_payload is not None:
        selected_clip_min = int(clip_length_min_payload)
        selected_clip_max = int(clip_length_max_payload)
        if selected_clip_min > selected_clip_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="clipLengthMinSeconds cannot be greater than clipLengthMaxSeconds.",
            )
    else:
        resolved_legacy_length = (
            int(clip_length_legacy_value)
            if clip_length_legacy_value is not None
            else min(_DEFAULT_ANALYZE_LEGACY_CLIP_SECONDS, clip_length_plan_max)
        )
        if (
            clip_length_legacy_value is not None
            and (
                resolved_legacy_length < _ANALYZE_CLIP_MIN_SECONDS
                or resolved_legacy_length > clip_length_plan_max
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Selected clip length is outside your plan limits. "
                    f"Allowed range: {_ANALYZE_CLIP_MIN_SECONDS}-{clip_length_plan_max} seconds."
                ),
            )
        selected_clip_min, selected_clip_max = _resolve_legacy_clip_range(
            clip_length_seconds=resolved_legacy_length,
            plan_max_seconds=clip_length_plan_max,
        )

    if (
        selected_clip_min < _ANALYZE_CLIP_MIN_SECONDS
        or selected_clip_max > clip_length_plan_max
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Selected clip length range is outside your plan limits. "
                f"Allowed range: {_ANALYZE_CLIP_MIN_SECONDS}-{clip_length_plan_max} seconds."
            ),
        )

    processing_start_seconds, processing_end_seconds = _normalize_processing_window(
        start_seconds=float(payload.processingStartSeconds),
        end_seconds=payload.processingEndSeconds,
        duration_seconds=analysis_duration_seconds,
    )
    processing_window_duration = max(0.0, processing_end_seconds - processing_start_seconds)
    if processing_window_duration < float(selected_clip_min):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Selected processing timeframe is shorter than the requested minimum clip length. "
                "Increase the timeframe or choose a shorter clip range."
            ),
        )
    billed_analysis_credits = calculate_video_analysis_cost(int(ceil(processing_window_duration)))

    max_clip_count = _max_clip_count_for_duration(int(processing_window_duration))
    if payload.numClips > max_clip_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many clips requested for this processing window. Maximum: {max_clip_count} "
                "(1 clip every 3 minutes)."
            ),
        )

    _raise_if_insufficient_credits(
        context=access_context,
        actor_user_id=user_id,
        required_credits=billed_analysis_credits,
    )

    existing_query = supabase.table("videos").select("id").eq("url", url_str)
    if access_context.workspace_team_id:
        existing_query = existing_query.eq("team_id", access_context.workspace_team_id)
    else:
        existing_query = existing_query.eq("user_id", user_id).is_("team_id", "null")
    existing = existing_query.limit(1).execute()
    raise_on_error(existing, "Failed to query existing video")
    created_video_for_request = False
    if existing.data and len(existing.data) > 0:
        video_id = existing.data[0]["id"]
        logger.info("Reusing existing video %s for analyze url", video_id)
    else:
        enforce_monthly_video_limit(user_id, supabase_client=supabase)
        # SECURITY: Always generate a new UUID for new videos. Never trust
        # a user-supplied videoId — it could reference another user's video
        # and the service-role upsert would overwrite it.
        video_id = str(uuid4())
        created_video_for_request = True

    logger.info(
        "Analyze request - user=%s  video=%s  url=%s  numClips=%d  clipRange=%d-%ds  "
        "window=%.2f-%.2f",
        user_id,
        video_id,
        url_str,
        payload.numClips,
        selected_clip_min,
        selected_clip_max,
        processing_start_seconds,
        processing_end_seconds,
    )
    logger.info(
        "Analyze request accepted - user=%s  url=%s  requiredCredits=%s  fullDurationCredits=%s",
        user_id,
        url_str,
        billed_analysis_credits,
        url_probe["analysis_credits"],
    )

    video_resp = (
        supabase.table("videos")
        .upsert(
            {
                "id": video_id,
                "user_id": user_id,
                "team_id": access_context.workspace_team_id,
                "billing_owner_user_id": access_context.billing_owner_user_id or user_id,
                "url": url_str,
                "status": "pending",
            },
            on_conflict="id",
        )
        .execute()
    )
    raise_on_error(video_resp, "Failed to upsert video")

    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "userId": user_id,
        "url": url_str,
        "numClips": payload.numClips,
        "clipLengthSeconds": selected_clip_max,
        "clipLengthMinSeconds": selected_clip_min,
        "clipLengthMaxSeconds": selected_clip_max,
        "processingStartSeconds": processing_start_seconds,
        "processingEndSeconds": processing_end_seconds,
        "extraPrompt": (payload.extraPrompt.strip() if payload.extraPrompt else None),
        "analysisCredits": billed_analysis_credits,
        "analysisDurationSeconds": analysis_duration_seconds,
        "sourceTitle": url_probe.get("video_title"),
        "sourceThumbnailUrl": url_probe.get("thumbnail_url"),
        "sourcePlatform": url_probe.get("platform"),
        "sourceExternalId": url_probe.get("external_id"),
        "sourceHasCaptions": url_probe.get("has_captions"),
        "sourceHasAudio": url_probe.get("has_audio"),
        "workspaceTeamId": access_context.workspace_team_id,
        "billingOwnerUserId": access_context.billing_owner_user_id or user_id,
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
                "billing_owner_user_id": access_context.billing_owner_user_id or user_id,
                "video_id": video_id,
                "type": "analyze_video",
                "status": "queued",
                "input_data": job_data,
            },
            on_conflict="id",
        )
        .execute()
    )
    raise_on_error(job_resp, "Failed to upsert job")

    def _cleanup_analyze_queue_full() -> None:
        if created_video_for_request:
            _best_effort_delete_video(video_id)

    enqueue_or_fail(
        queue_name=_video_queue_for_context(access_context),
        task_path=ANALYZE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="analyze_video",
        video_id=video_id,
        on_queue_full_cleanup=_cleanup_analyze_queue_full,
    )

    return AnalyzeVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@router.post("/credits/cost", response_model=CreditsCostByUrlResponse)
@limiter.limit("15/minute")
@limiter.limit("10/minute", key_func=get_user_rate_key)
async def get_credit_cost_from_url(
    request: Request,
    payload: CreditsCostByUrlRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CreditsCostByUrlResponse:
    """Validate URL and return analysis cost plus first-generation estimate."""
    url_str = str(payload.url)
    requested_clip_count = int(payload.numClips)
    user_id = current_user.id
    access_context = get_user_access_context(user_id, supabase_client=supabase)
    # Suggested clips created from AI analysis are free on first generation.
    clip_generation_credits_per_clip = 0
    smart_cleanup_surcharge_per_clip = 0
    try:
        probe = await asyncio.wait_for(
            asyncio.to_thread(_probe_credit_cost_for_url, url_str),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Video URL probe timed out. Please try again.",
        )
    max_analysis_duration_seconds = access_context.max_analysis_duration_seconds
    analysis_duration_seconds = int(probe.get("duration_seconds") or 0)
    duration_limit_exceeded = not is_analysis_duration_allowed(
        context=access_context,
        duration_seconds=analysis_duration_seconds,
    )
    balance_owner_user_id = access_context.billing_owner_user_id or user_id
    current_balance = (
        get_team_wallet_balance(access_context.workspace_team_id)
        if access_context.charge_source == "team_wallet" and access_context.workspace_team_id
        else get_credit_balance(balance_owner_user_id)
    )

    if not probe["valid_url"]:
        return CreditsCostByUrlResponse(
            valid_url=False,
            analysisCredits=0,
            totalCredits=0,
            requestedClipCount=requested_clip_count,
            clipGenerationCreditsPerClip=clip_generation_credits_per_clip,
            estimatedGenerationCredits=0,
            smartCleanupSurchargePerClip=smart_cleanup_surcharge_per_clip,
            estimatedTotalCredits=0,
            analysisDurationSeconds=analysis_duration_seconds,
            maxAnalysisDurationSeconds=max_analysis_duration_seconds,
            durationLimitExceeded=duration_limit_exceeded,
            currentBalance=current_balance,
            hasEnoughCredits=False,
            hasEnoughCreditsForEstimatedTotal=False,
            videoTitle=probe.get("video_title"),
            thumbnailUrl=probe.get("thumbnail_url"),
            platform=probe.get("platform"),
            detectedLanguage=probe.get("detected_language"),
        )

    analysis_credits = int(probe["analysis_credits"])
    estimated_generation_credits = requested_clip_count * clip_generation_credits_per_clip
    estimated_total_credits = analysis_credits + estimated_generation_credits
    has_enough_credits = current_balance >= analysis_credits and not duration_limit_exceeded
    has_enough_credits_for_estimated_total = (
        current_balance >= estimated_total_credits and not duration_limit_exceeded
    )
    return CreditsCostByUrlResponse(
        valid_url=True,
        analysisCredits=analysis_credits,
        totalCredits=analysis_credits,
        requestedClipCount=requested_clip_count,
        clipGenerationCreditsPerClip=clip_generation_credits_per_clip,
        estimatedGenerationCredits=estimated_generation_credits,
        smartCleanupSurchargePerClip=smart_cleanup_surcharge_per_clip,
        estimatedTotalCredits=estimated_total_credits,
        analysisDurationSeconds=analysis_duration_seconds,
        maxAnalysisDurationSeconds=max_analysis_duration_seconds,
        durationLimitExceeded=duration_limit_exceeded,
        currentBalance=current_balance,
        hasEnoughCredits=has_enough_credits,
        hasEnoughCreditsForEstimatedTotal=has_enough_credits_for_estimated_total,
        videoTitle=probe.get("video_title"),
        thumbnailUrl=probe.get("thumbnail_url"),
        platform=probe.get("platform"),
        detectedLanguage=probe.get("detected_language"),
    )

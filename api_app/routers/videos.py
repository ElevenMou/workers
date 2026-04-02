"""Video analysis and credit-cost endpoints."""

import asyncio
from datetime import datetime, timezone
from math import ceil
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from api_app.auth import get_user_rate_key

from api_app.access_rules import (
    UserAccessContext,
    enforce_analysis_duration_limit,
    enforce_custom_clip_access,
    enforce_monthly_video_limit,
    enforce_processing_access_rules,
    get_user_access_context,
    is_analysis_duration_allowed,
)
from api_app.auth import AuthenticatedUser, get_current_user

from api_app.constants import ANALYZE_TASK_PATH, GENERATE_TASK_PATH, SPLIT_VIDEO_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    AnalyzeVideoRequest,
    AnalyzeVideoResponse,
    BatchGeneratePreparedClipsRequest,
    BatchGeneratePreparedClipsResponse,
    BatchSplitVideoRequest,
    BatchSplitVideoResponse,
    CreditsCostByUrlRequest,
    CreditsCostByUrlResponse,
)
from api_app.routers.clips import (
    _ACTIVE_GENERATE_JOB_STATUSES,
    _clip_queue_for_context,
    _enforce_smart_cleanup_access,
    _raise_if_insufficient_clip_generation_credits,
)
from api_app.rate_limit import limiter
from api_app.state import logger, whisper_ready
from config import (
    VIDEO_JOB_TIMEOUT,
    WHISPER_FALLBACK_JOB_TIMEOUT_MAX_SECONDS,
    WHISPER_FALLBACK_JOB_TIMEOUT_MULTIPLIER,
    WHISPER_FALLBACK_JOB_TIMEOUT_PADDING_SECONDS,
    calculate_clip_generation_cost,
    calculate_custom_clip_generation_cost,
    calculate_video_analysis_cost,
)
from services.clips.quality_policy import resolve_clip_quality_policy
from services.clips.render_profiles import (
    DEFAULT_DELIVERY_PROFILE,
    DEFAULT_MASTER_PROFILE,
    DEFAULT_SOURCE_PROFILE,
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
_ACTIVE_ANALYZE_JOB_STATUSES = ("queued", "processing", "retrying")
_ANALYZE_CLIP_MIN_SECONDS = 10
_DEFAULT_ANALYZE_LEGACY_CLIP_SECONDS = 90
_URL_PROBE_TIMEOUT_SECONDS = 45.0


def _best_effort_delete_video(video_id: str):
    try:
        supabase.table("videos").delete().eq("id", video_id).execute()
    except Exception as exc:
        logger.warning("Failed cleanup delete for video %s: %s", video_id, exc)


def _best_effort_supersede_prior_analyze_jobs(
    *,
    video_id: str,
    keep_job_id: str,
    workspace_team_id: str | None,
    actor_user_id: str,
):
    reason = (
        "Superseded by a newer video analysis request for the same source. "
        "This job exited without charging credits."
    )
    now_utc = datetime.now(timezone.utc).isoformat()

    try:
        active_job_query = (
            supabase.table("jobs")
            .select("id,status")
            .eq("video_id", video_id)
            .eq("type", "analyze_video")
            .in_("status", list(_ACTIVE_ANALYZE_JOB_STATUSES))
        )
        if workspace_team_id:
            active_job_query = active_job_query.eq("team_id", workspace_team_id)
        else:
            active_job_query = active_job_query.eq("user_id", actor_user_id).is_("team_id", "null")

        active_job_resp = (
            active_job_query
            .order("created_at", desc=True)
            .order("id", desc=True)
            .execute()
        )
        raise_on_error(active_job_resp, "Failed to load active analyze jobs for supersede")
        active_jobs = list(active_job_resp.data or [])
    except Exception as exc:
        logger.warning(
            "Failed to inspect prior analyze jobs for video %s: %s",
            video_id,
            exc,
        )
        return

    superseded_count = 0
    for row in active_jobs:
        prior_job_id = str(row.get("id") or "").strip()
        if not prior_job_id or prior_job_id == keep_job_id:
            continue

        try:
            resp = (
                supabase.table("jobs")
                .update(
                    {
                        "status": "failed",
                        "progress": 0,
                        "completed_at": now_utc,
                        "error_message": reason,
                        "result_data": {
                            "stage": "superseded",
                            "detail_key": "superseded_by_newer_request",
                            "video_id": video_id,
                        },
                    }
                )
                .eq("id", prior_job_id)
                .execute()
            )
            raise_on_error(resp, f"Failed to mark prior analyze job {prior_job_id} superseded")
            superseded_count += 1
        except Exception as exc:
            logger.warning(
                "Failed to mark prior analyze job %s superseded for video %s: %s",
                prior_job_id,
                video_id,
                exc,
            )

    if superseded_count:
        logger.info(
            "Superseded %d prior analyze job(s) for video=%s in favor of job=%s",
            superseded_count,
            video_id,
            keep_job_id,
        )


def _video_queue_for_context(context: UserAccessContext) -> str:
    return _PRIORITY_VIDEO_QUEUE if context.priority_processing else _STANDARD_VIDEO_QUEUE


def _compute_video_job_timeout_seconds(
    *,
    duration_seconds: int,
    source_has_captions: bool | None,
    source_has_audio: bool | None,
) -> int:
    """Return a queue timeout sized for likely Whisper fallback work."""
    default_timeout = int(VIDEO_JOB_TIMEOUT)
    if source_has_captions:
        return default_timeout
    if source_has_audio is False:
        return default_timeout

    duration = max(0, int(duration_seconds))
    if duration <= 0:
        return default_timeout

    whisper_timeout = int(
        ceil(duration * float(WHISPER_FALLBACK_JOB_TIMEOUT_MULTIPLIER))
    ) + int(WHISPER_FALLBACK_JOB_TIMEOUT_PADDING_SECONDS)
    return min(
        int(WHISPER_FALLBACK_JOB_TIMEOUT_MAX_SECONDS),
        max(default_timeout, whisper_timeout),
    )


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


def _max_clip_count_for_duration(
    duration_seconds: int,
    *,
    selected_clip_max_seconds: int,
) -> int:
    seconds = max(int(duration_seconds), 0)
    clip_max = max(_ANALYZE_CLIP_MIN_SECONDS, int(selected_clip_max_seconds))
    return min(20, max(1, seconds // clip_max))


def _load_workspace_video(
    *,
    video_id: str,
    access_context: UserAccessContext,
    actor_user_id: str,
) -> dict | None:
    response = (
        supabase.table("videos")
        .select(
            "id,user_id,team_id,url,title,duration_seconds,thumbnail_url,platform,external_id,"
            "raw_video_path,raw_video_storage_path,transcript,status"
        )
        .eq("id", video_id)
        .limit(1)
        .execute()
    )
    raise_on_error(response, "Failed to load video")
    rows = response.data or []
    if not rows:
        return None

    video = rows[0]
    if access_context.workspace_team_id:
        if video.get("team_id") != access_context.workspace_team_id:
            return None
    else:
        if video.get("team_id") or video.get("user_id") != actor_user_id:
            return None
    return video


def _load_pending_batch_split_clips(
    *,
    video_id: str,
    access_context: UserAccessContext,
    actor_user_id: str,
) -> list[dict]:
    clip_query = (
        supabase.table("clips")
        .select(
            "id,video_id,user_id,team_id,billing_owner_user_id,start_time,end_time,status,"
            "storage_path,ai_score,transcript_excerpt,layout_id,title"
        )
        .eq("video_id", video_id)
        .eq("origin", "batch_split")
        .eq("status", "pending")
        .order("created_at", desc=False)
    )
    if access_context.workspace_team_id:
        clip_query = clip_query.eq("team_id", access_context.workspace_team_id)
        if access_context.workspace_role != "owner":
            clip_query = clip_query.eq("user_id", actor_user_id)
    else:
        clip_query = clip_query.eq("user_id", actor_user_id).is_("team_id", "null")

    clip_response = clip_query.execute()
    raise_on_error(clip_response, "Failed to load prepared split clips")
    return list(clip_response.data or [])


def _load_active_generate_jobs_for_clips(
    *,
    clip_ids: list[str],
    access_context: UserAccessContext,
    actor_user_id: str,
) -> set[str]:
    if not clip_ids:
        return set()

    job_query = (
        supabase.table("jobs")
        .select("clip_id")
        .eq("type", "generate_clip")
        .in_("status", list(_ACTIVE_GENERATE_JOB_STATUSES))
        .in_("clip_id", clip_ids)
        .order("created_at", desc=True)
    )
    if access_context.workspace_team_id:
        job_query = job_query.eq("team_id", access_context.workspace_team_id)
    else:
        job_query = job_query.eq("user_id", actor_user_id).is_("team_id", "null")

    job_response = job_query.execute()
    raise_on_error(job_response, "Failed to load active generation jobs for prepared clips")
    return {
        str(row.get("clip_id") or "").strip()
        for row in list(job_response.data or [])
        if str(row.get("clip_id") or "").strip()
    }


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
            timeout=_URL_PROBE_TIMEOUT_SECONDS,
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

    max_clip_count = _max_clip_count_for_duration(
        int(processing_window_duration),
        selected_clip_max_seconds=selected_clip_max,
    )
    if payload.numClips > max_clip_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many clips requested for this processing window. Maximum: {max_clip_count} "
                f"(based on your selected {selected_clip_max}s max clip length)."
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
        "sourceDetectedLanguage": url_probe.get("detected_language"),
        "sourceHasCaptions": url_probe.get("has_captions"),
        "sourceHasAudio": url_probe.get("has_audio"),
        "workspaceTeamId": access_context.workspace_team_id,
        "billingOwnerUserId": access_context.billing_owner_user_id or user_id,
        "chargeSource": access_context.charge_source,
        "workspaceRole": access_context.workspace_role,
    }
    job_timeout_seconds = _compute_video_job_timeout_seconds(
        duration_seconds=analysis_duration_seconds,
        source_has_captions=url_probe.get("has_captions"),
        source_has_audio=url_probe.get("has_audio"),
    )

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
        job_timeout_seconds=job_timeout_seconds,
        job_id=job_id,
        user_id=user_id,
        job_type="analyze_video",
        video_id=video_id,
        on_queue_full_cleanup=_cleanup_analyze_queue_full,
    )
    _best_effort_supersede_prior_analyze_jobs(
        video_id=video_id,
        keep_job_id=job_id,
        workspace_team_id=access_context.workspace_team_id,
        actor_user_id=user_id,
    )

    return AnalyzeVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@router.post("/videos/batch-split", response_model=BatchSplitVideoResponse)
@limiter.limit("10/minute")
@limiter.limit("5/minute", key_func=get_user_rate_key)
async def batch_split_video(
    request: Request,
    payload: BatchSplitVideoRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> BatchSplitVideoResponse:
    """Queue a fixed-window batch split job for a direct URL or existing video."""
    user_id = current_user.id
    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)
    enforce_custom_clip_access(context=access_context)

    if bool(payload.videoId) == bool(payload.url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of videoId or url.",
        )

    segment_length_seconds = int(payload.segmentLengthSeconds)
    if segment_length_seconds > int(access_context.max_clip_duration_seconds):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Selected segment length is outside your plan limits. "
                f"Allowed maximum: {int(access_context.max_clip_duration_seconds)} seconds."
            ),
        )

    video_row: dict | None = None
    created_video_for_request = False
    source_probe: dict | None = None
    if payload.videoId:
        video_row = _load_workspace_video(
            video_id=str(payload.videoId),
            access_context=access_context,
            actor_user_id=user_id,
        )
        if video_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found in the active workspace.",
            )
        video_id = str(video_row["id"])
        source_url = str(video_row.get("url") or "").strip()
        if not source_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This video no longer has a valid source URL.",
            )
    else:
        source_url = str(payload.url)
        existing_query = supabase.table("videos").select("id").eq("url", source_url)
        if access_context.workspace_team_id:
            existing_query = existing_query.eq("team_id", access_context.workspace_team_id)
        else:
            existing_query = existing_query.eq("user_id", user_id).is_("team_id", "null")
        existing_response = existing_query.limit(1).execute()
        raise_on_error(existing_response, "Failed to query existing split-video source")
        existing_rows = existing_response.data or []
        if existing_rows:
            video_id = str(existing_rows[0]["id"])
            video_row = _load_workspace_video(
                video_id=video_id,
                access_context=access_context,
                actor_user_id=user_id,
            )
        else:
            enforce_monthly_video_limit(user_id, supabase_client=supabase)
            video_id = str(uuid4())
            created_video_for_request = True

        try:
            source_probe = await asyncio.wait_for(
                asyncio.to_thread(_probe_credit_cost_for_url, source_url),
                timeout=_URL_PROBE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Video URL probe timed out. Please try again.",
            )
        if not source_probe["valid_url"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Video URL cannot be split. Ensure it is downloadable and has "
                    "captions (or audio for transcription)."
                ),
            )

    active_job_query = (
        supabase.table("jobs")
        .select("id,status")
        .eq("video_id", video_id)
        .eq("type", "split_video")
        .in_("status", list(_ACTIVE_ANALYZE_JOB_STATUSES))
    )
    if access_context.workspace_team_id:
        active_job_query = active_job_query.eq("team_id", access_context.workspace_team_id)
    else:
        active_job_query = active_job_query.eq("user_id", user_id).is_("team_id", "null")
    active_job_response = (
        active_job_query
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    raise_on_error(active_job_response, "Failed to check active split-video job")
    active_job_rows = active_job_response.data or []
    if active_job_rows:
        active_job = active_job_rows[0]
        return BatchSplitVideoResponse(
            jobId=str(active_job["id"]),
            videoId=video_id,
            status=str(active_job.get("status") or "queued"),
        )

    if video_row and (
        not int(video_row.get("duration_seconds") or 0)
        or not str(video_row.get("platform") or "").strip()
    ):
        try:
            source_probe = await asyncio.wait_for(
                asyncio.to_thread(_probe_credit_cost_for_url, source_url),
                timeout=_URL_PROBE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Video URL probe timed out. Please try again.",
            )
        if source_probe and not source_probe["valid_url"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This video source is no longer valid for processing.",
            )

    source_duration_seconds = int(
        (source_probe or {}).get("duration_seconds")
        or (video_row or {}).get("duration_seconds")
        or 0
    )
    if source_duration_seconds <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine the source video duration for batch split.",
        )

    estimated_part_count = min(
        20,
        max(1, int(ceil(source_duration_seconds / max(1, segment_length_seconds)))),
    )
    expected_generation_credits = int(
        estimated_part_count
        * calculate_custom_clip_generation_cost(False, access_context.tier)
    )

    billing_owner_user_id = access_context.billing_owner_user_id or user_id
    source_title = (
        str((source_probe or {}).get("video_title") or "").strip()
        or str((video_row or {}).get("title") or "").strip()
        or None
    )
    source_thumbnail_url = (
        str((source_probe or {}).get("thumbnail_url") or "").strip()
        or str((video_row or {}).get("thumbnail_url") or "").strip()
        or None
    )
    source_platform = (
        str((source_probe or {}).get("platform") or "").strip().lower()
        or str((video_row or {}).get("platform") or "").strip().lower()
        or None
    )
    source_external_id = (
        str((source_probe or {}).get("external_id") or "").strip()
        or str((video_row or {}).get("external_id") or "").strip()
        or None
    )
    source_detected_language = (
        str((source_probe or {}).get("detected_language") or "").strip() or None
    )
    source_has_audio = (
        source_probe.get("has_audio") if source_probe is not None else None
    )
    source_has_captions = (
        source_probe.get("has_captions") if source_probe is not None else None
    )

    video_upsert_payload = {
        "id": video_id,
        "user_id": user_id,
        "team_id": access_context.workspace_team_id,
        "billing_owner_user_id": billing_owner_user_id,
        "url": source_url,
        "status": (
            "pending"
            if created_video_for_request
            else str((video_row or {}).get("status") or "pending")
        ),
    }
    if source_title:
        video_upsert_payload["title"] = source_title
    if source_thumbnail_url:
        video_upsert_payload["thumbnail_url"] = source_thumbnail_url
    if source_platform:
        video_upsert_payload["platform"] = source_platform
    if source_external_id:
        video_upsert_payload["external_id"] = source_external_id
    if source_duration_seconds > 0:
        video_upsert_payload["duration_seconds"] = source_duration_seconds

    video_response = (
        supabase.table("videos")
        .upsert(video_upsert_payload, on_conflict="id")
        .execute()
    )
    raise_on_error(video_response, "Failed to upsert split-video source")

    job_id = str(uuid4())
    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "userId": user_id,
        "url": source_url,
        "layoutId": payload.layoutId,
        "segmentLengthSeconds": segment_length_seconds,
        "expectedPartCount": estimated_part_count,
        "expectedGenerationCredits": expected_generation_credits,
        "maxParts": 20,
        "clipRetentionDays": access_context.clip_retention_days,
        "workspaceTeamId": access_context.workspace_team_id,
        "billingOwnerUserId": billing_owner_user_id,
        "chargeSource": access_context.charge_source,
        "workspaceRole": access_context.workspace_role,
        "subscriptionTier": access_context.tier,
        "priorityProcessing": access_context.priority_processing,
        "sourceTitle": source_title,
        "sourceThumbnailUrl": source_thumbnail_url,
        "sourcePlatform": source_platform,
        "sourceExternalId": source_external_id,
        "sourceDetectedLanguage": source_detected_language,
        "sourceHasCaptions": source_has_captions,
        "sourceHasAudio": source_has_audio,
        "sourceDurationSeconds": source_duration_seconds,
    }
    job_response = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": user_id,
                "team_id": access_context.workspace_team_id,
                "billing_owner_user_id": billing_owner_user_id,
                "video_id": video_id,
                "type": "split_video",
                "status": "queued",
                "input_data": job_data,
            },
            on_conflict="id",
        )
        .execute()
    )
    raise_on_error(job_response, "Failed to upsert split-video job")
    job_timeout_seconds = _compute_video_job_timeout_seconds(
        duration_seconds=source_duration_seconds,
        source_has_captions=source_has_captions,
        source_has_audio=source_has_audio,
    )

    def _cleanup_split_queue_full() -> None:
        if created_video_for_request:
            _best_effort_delete_video(video_id)

    enqueue_or_fail(
        queue_name=_video_queue_for_context(access_context),
        task_path=SPLIT_VIDEO_TASK_PATH,
        job_data=job_data,
        job_timeout_seconds=job_timeout_seconds,
        job_id=job_id,
        user_id=user_id,
        job_type="split_video",
        video_id=video_id,
        on_queue_full_cleanup=_cleanup_split_queue_full,
    )

    return BatchSplitVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@router.post(
    "/videos/batch-generate-prepared-clips",
    response_model=BatchGeneratePreparedClipsResponse,
)
@limiter.limit("10/minute")
@limiter.limit("5/minute", key_func=get_user_rate_key)
def batch_generate_prepared_clips(
    request: Request,
    payload: BatchGeneratePreparedClipsRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> BatchGeneratePreparedClipsResponse:
    """Queue generation jobs for eligible prepared split clips on one video."""
    user_id = current_user.id
    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)
    smart_cleanup_enabled = bool(payload.smartCleanupEnabled)
    _enforce_smart_cleanup_access(
        context=access_context,
        smart_cleanup_enabled=smart_cleanup_enabled,
    )

    video_id = str(payload.videoId or "").strip()
    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="videoId is required.",
        )

    video_row = _load_workspace_video(
        video_id=video_id,
        access_context=access_context,
        actor_user_id=user_id,
    )
    if video_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found in the active workspace.",
        )

    pending_clips = _load_pending_batch_split_clips(
        video_id=video_id,
        access_context=access_context,
        actor_user_id=user_id,
    )
    if not pending_clips:
        return BatchGeneratePreparedClipsResponse(
            videoId=video_id,
            queuedCount=0,
            skippedCount=0,
            jobIds=[],
        )

    active_clip_ids = _load_active_generate_jobs_for_clips(
        clip_ids=[
            str(clip.get("id") or "").strip()
            for clip in pending_clips
            if str(clip.get("id") or "").strip()
        ],
        access_context=access_context,
        actor_user_id=user_id,
    )

    eligible_clips: list[dict] = []
    skipped_count = 0
    max_clip_duration_seconds = int(access_context.max_clip_duration_seconds)
    for clip in pending_clips:
        clip_id = str(clip.get("id") or "").strip()
        if not clip_id:
            skipped_count += 1
            continue

        if clip_id in active_clip_ids:
            skipped_count += 1
            continue

        clip_duration = float(clip.get("end_time") or 0) - float(clip.get("start_time") or 0)
        if clip_duration > float(max_clip_duration_seconds):
            skipped_count += 1
            continue

        eligible_clips.append(clip)

    if not eligible_clips:
        return BatchGeneratePreparedClipsResponse(
            videoId=video_id,
            queuedCount=0,
            skippedCount=skipped_count,
            jobIds=[],
        )

    generation_credits_per_clip = calculate_clip_generation_cost(
        smart_cleanup_enabled,
        access_context.tier,
    )
    total_required_credits = len(eligible_clips) * int(generation_credits_per_clip)
    _raise_if_insufficient_clip_generation_credits(
        context=access_context,
        actor_user_id=user_id,
        required_credits=total_required_credits,
    )

    queued_job_ids: list[str] = []
    billing_owner_user_id = access_context.billing_owner_user_id or user_id
    for clip in eligible_clips:
        clip_id = str(clip["id"])
        clip_duration = float(clip["end_time"]) - float(clip["start_time"])
        quality_policy = resolve_clip_quality_policy(
            tier=access_context.tier,
            clip_duration_seconds=clip_duration,
            requested_output_quality=None,
        )
        output_quality_override = None
        job_id = str(uuid4())
        job_data = {
            "jobId": job_id,
            "clipId": clip_id,
            "userId": user_id,
            "layoutId": clip.get("layout_id"),
            "smartCleanupEnabled": smart_cleanup_enabled,
            "generationCredits": int(generation_credits_per_clip),
            "clipRetentionDays": access_context.clip_retention_days,
            "workspaceTeamId": access_context.workspace_team_id,
            "billingOwnerUserId": billing_owner_user_id,
            "chargeSource": access_context.charge_source,
            "workspaceRole": access_context.workspace_role,
            "subscriptionTier": access_context.tier,
            "sourceMaxHeight": quality_policy.get("source_max_height"),
            "sourceProfile": DEFAULT_SOURCE_PROFILE,
            "masterProfile": DEFAULT_MASTER_PROFILE,
            "deliveryProfile": DEFAULT_DELIVERY_PROFILE,
            "outputQualityOverride": output_quality_override,
            "qualityPolicyProfile": quality_policy.get("profile"),
        }

        job_response = (
            supabase.table("jobs")
            .upsert(
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
                },
                on_conflict="id",
            )
            .execute()
        )
        raise_on_error(job_response, f"Failed to queue prepared clip job for {clip_id}")

        enqueue_or_fail(
            queue_name=_clip_queue_for_context(access_context),
            task_path=GENERATE_TASK_PATH,
            job_data=job_data,
            job_id=job_id,
            user_id=user_id,
            job_type="generate_clip",
            video_id=video_id,
            clip_id=clip_id,
        )
        queued_job_ids.append(job_id)

    return BatchGeneratePreparedClipsResponse(
        videoId=video_id,
        queuedCount=len(queued_job_ids),
        skippedCount=skipped_count,
        jobIds=queued_job_ids,
    )


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
            timeout=_URL_PROBE_TIMEOUT_SECONDS,
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

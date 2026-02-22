"""Video analysis and credit-cost endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api_app.access_rules import (
    UserAccessContext,
    enforce_analysis_duration_limit,
    enforce_monthly_video_limit,
    enforce_processing_access_rules,
    get_user_access_context,
    is_analysis_duration_allowed,
)
from api_app.auth import AuthenticatedUser, get_current_user

limiter = Limiter(key_func=get_remote_address)
from api_app.constants import ANALYZE_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    AnalyzeVideoRequest,
    AnalyzeVideoResponse,
    CreditsCostByUrlRequest,
    CreditsCostByUrlResponse,
)
from api_app.state import logger, whisper_ready
from config import calculate_video_analysis_cost
from services.video_downloader import VideoDownloader
from utils.supabase_client import get_credit_balance, has_sufficient_credits, supabase

router = APIRouter()
_STANDARD_VIDEO_QUEUE = "video-processing"
_PRIORITY_VIDEO_QUEUE = "video-processing-priority"


def _video_queue_for_context(context: UserAccessContext) -> str:
    return _PRIORITY_VIDEO_QUEUE if context.priority_processing else _STANDARD_VIDEO_QUEUE


def _probe_credit_cost_for_url(url_str: str) -> dict:
    downloader = VideoDownloader()

    try:
        probe = downloader.probe_url(url_str)
    except Exception as exc:
        logger.warning("URL validation failed for %s: %s", url_str, exc)
        return {
            "valid_url": False,
            "analysis_credits": 0,
            "duration_seconds": 0,
        }

    duration_seconds = int(probe.get("duration_seconds") or 0)
    can_download = bool(probe.get("can_download"))
    has_source_captions = bool(probe.get("has_captions"))
    has_audio = bool(probe.get("has_audio"))
    can_use_whisper = has_audio and whisper_ready()
    has_captions = has_source_captions or can_use_whisper

    valid_url = can_download and has_captions and duration_seconds > 0
    if not valid_url:
        return {
            "valid_url": False,
            "analysis_credits": 0,
            "duration_seconds": duration_seconds,
        }

    analysis_credits = calculate_video_analysis_cost(duration_seconds)
    return {
        "valid_url": True,
        "analysis_credits": analysis_credits,
        "duration_seconds": duration_seconds,
    }


def _raise_if_insufficient_credits(*, user_id: str, required_credits: int):
    if required_credits <= 0:
        return

    if has_sufficient_credits(user_id=user_id, amount=required_credits):
        return

    balance = get_credit_balance(user_id)
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=(
            "Insufficient credits for analysis. "
            f"Required: {required_credits}, available: {balance}."
        ),
    )


@router.post("/videos/analyze", response_model=AnalyzeVideoResponse)
@limiter.limit("10/minute")
def analyze_video(
    request: Request,
    payload: AnalyzeVideoRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AnalyzeVideoResponse:
    """Create and enqueue a video analysis job."""
    job_id = str(uuid4())
    url_str = str(payload.url)
    user_id = current_user.id

    # Fast-fail if user has no credits at all (before URL probing).
    _raise_if_insufficient_credits(user_id=user_id, required_credits=1)

    access_context = enforce_processing_access_rules(user_id, supabase_client=supabase)

    # Enforce validation gate equivalent to /credits/cost before job enqueue.
    url_probe = _probe_credit_cost_for_url(url_str)
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

    _raise_if_insufficient_credits(
        user_id=user_id,
        required_credits=int(url_probe["analysis_credits"]),
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
        logger.info("Reusing existing video %s for analyze url", video_id)
    else:
        enforce_monthly_video_limit(user_id, supabase_client=supabase)
        # SECURITY: Always generate a new UUID for new videos. Never trust
        # a user-supplied videoId — it could reference another user's video
        # and the service-role upsert would overwrite it.
        video_id = str(uuid4())

    logger.info(
        "Analyze request - user=%s  video=%s  url=%s  numClips=%d",
        user_id,
        video_id,
        url_str,
        payload.numClips,
    )
    logger.info(
        "Analyze request accepted - user=%s  url=%s  analysisCredits=%s",
        user_id,
        url_str,
        url_probe["analysis_credits"],
    )

    video_resp = (
        supabase.table("videos")
        .upsert(
            {
                "id": video_id,
                "user_id": user_id,
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
        "analysisCredits": int(url_probe["analysis_credits"]),
        "analysisDurationSeconds": analysis_duration_seconds,
    }

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": user_id,
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

    enqueue_or_fail(
        queue_name=_video_queue_for_context(access_context),
        task_path=ANALYZE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="analyze_video",
        video_id=video_id,
    )

    return AnalyzeVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@router.post("/credits/cost", response_model=CreditsCostByUrlResponse)
@limiter.limit("15/minute")
def get_credit_cost_from_url(
    request: Request,
    payload: CreditsCostByUrlRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CreditsCostByUrlResponse:
    """Validate URL for clipping and return analysis credit cost only."""
    url_str = str(payload.url)
    user_id = current_user.id
    access_context = get_user_access_context(user_id, supabase_client=supabase)
    probe = _probe_credit_cost_for_url(url_str)
    max_analysis_duration_seconds = access_context.max_analysis_duration_seconds
    analysis_duration_seconds = int(probe.get("duration_seconds") or 0)
    duration_limit_exceeded = not is_analysis_duration_allowed(
        context=access_context,
        duration_seconds=analysis_duration_seconds,
    )

    if not probe["valid_url"]:
        return CreditsCostByUrlResponse(
            valid_url=False,
            analysisCredits=0,
            totalCredits=0,
            analysisDurationSeconds=analysis_duration_seconds,
            maxAnalysisDurationSeconds=max_analysis_duration_seconds,
            durationLimitExceeded=duration_limit_exceeded,
            currentBalance=get_credit_balance(user_id),
            hasEnoughCredits=False,
        )

    analysis_credits = int(probe["analysis_credits"])
    balance = get_credit_balance(user_id)
    has_enough_credits = balance >= analysis_credits and not duration_limit_exceeded
    return CreditsCostByUrlResponse(
        valid_url=True,
        analysisCredits=analysis_credits,
        totalCredits=analysis_credits,
        analysisDurationSeconds=analysis_duration_seconds,
        maxAnalysisDurationSeconds=max_analysis_duration_seconds,
        durationLimitExceeded=duration_limit_exceeded,
        currentBalance=balance,
        hasEnoughCredits=has_enough_credits,
    )

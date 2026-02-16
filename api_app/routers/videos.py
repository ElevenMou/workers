"""Video analysis and credit-cost endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends

from api_app.auth import AuthenticatedUser, get_current_user
from api_app.constants import ANALYZE_TASK_PATH
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    AnalyzeVideoRequest,
    AnalyzeVideoResponse,
    CreditsCostByUrlRequest,
    CreditsCostByUrlResponse,
)
from api_app.state import logger, whisper_ready
from config import CREDIT_COST_CLIP_GENERATION, calculate_video_analysis_cost
from services.video_downloader import VideoDownloader
from utils.supabase_client import supabase

router = APIRouter()


@router.post("/videos/analyze", response_model=AnalyzeVideoResponse)
def analyze_video(
    payload: AnalyzeVideoRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AnalyzeVideoResponse:
    """Create and enqueue a video analysis job."""
    job_id = str(uuid4())
    url_str = str(payload.url)
    user_id = current_user.id

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
        video_id = payload.videoId or str(uuid4())

    logger.info(
        "Analyze request - user=%s  video=%s  url=%s  numClips=%d",
        user_id,
        video_id,
        url_str,
        payload.numClips,
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
        queue_name="video-processing",
        task_path=ANALYZE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="analyze_video",
        video_id=video_id,
    )

    return AnalyzeVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@router.post("/credits/cost", response_model=CreditsCostByUrlResponse)
def get_credit_cost_from_url(
    payload: CreditsCostByUrlRequest,
) -> CreditsCostByUrlResponse:
    """Validate URL for clipping and return analysis + generation credit costs."""
    url_str = str(payload.url)
    downloader = VideoDownloader()

    try:
        probe = downloader.probe_url(url_str)
    except Exception as exc:
        logger.warning("URL validation failed for cost endpoint (%s): %s", url_str, exc)
        return CreditsCostByUrlResponse(
            valid_url=False,
            analysisCredits=0,
            clipGenerationCredits=0,
            totalCredits=0,
        )

    duration_seconds = int(probe.get("duration_seconds") or 0)
    can_download = bool(probe.get("can_download"))
    has_source_captions = bool(probe.get("has_captions"))
    has_audio = bool(probe.get("has_audio"))
    can_use_whisper = has_audio and whisper_ready()
    has_captions = has_source_captions or can_use_whisper

    valid_url = can_download and has_captions and duration_seconds > 0
    if not valid_url:
        return CreditsCostByUrlResponse(
            valid_url=False,
            analysisCredits=0,
            clipGenerationCredits=0,
            totalCredits=0,
        )

    analysis_credits = calculate_video_analysis_cost(duration_seconds)
    clip_generation_credits = CREDIT_COST_CLIP_GENERATION
    return CreditsCostByUrlResponse(
        valid_url=True,
        analysisCredits=analysis_credits,
        clipGenerationCredits=clip_generation_credits,
        totalCredits=analysis_credits + clip_generation_credits,
    )

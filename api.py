import logging
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from config import CREDIT_COST_CLIP_GENERATION, calculate_video_analysis_cost, validate_env
from services.video_downloader import VideoDownloader
from utils.redis_client import enqueue_job
from utils.supabase_client import assert_response_ok, supabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.api")

app = FastAPI(title="Clipry Workers API", version="1.0.0")
validate_env()

_WHISPER_READY: bool | None = None

# Use string paths so the API process does not eagerly import heavy worker deps
ANALYZE_TASK_PATH = "tasks.analyze_video.analyze_video_task"
GENERATE_TASK_PATH = "tasks.generate_clip.generate_clip_task"
CUSTOM_CLIP_TASK_PATH = "tasks.custom_clip.custom_clip_task"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeVideoRequest(BaseModel):
    userId: str = Field(..., description="Supabase user id")
    url: HttpUrl
    numClips: int = Field(default=5, ge=1, le=20)
    videoId: Optional[str] = Field(
        default=None, description="Optional existing video id to reuse"
    )


class AnalyzeVideoResponse(BaseModel):
    jobId: str
    videoId: str
    status: str


class GenerateClipRequest(BaseModel):
    userId: str = Field(..., description="Supabase user id")
    clipId: str = Field(..., description="Clip id to generate")
    layoutId: Optional[str] = Field(
        default=None,
        description="UUID of a saved layout from the layouts table. "
        "The worker loads all generation settings (background, video, "
        "title, captions, quality) from this layout. "
        "When omitted, built-in defaults are used.",
    )


class CustomClipRequest(BaseModel):
    userId: str = Field(..., description="Supabase user id")
    url: HttpUrl = Field(..., description="Video URL (YouTube, etc.)")
    startTime: float = Field(..., ge=0, description="Clip start time in seconds")
    endTime: float = Field(..., gt=0, description="Clip end time in seconds")
    title: str = Field(..., min_length=1, max_length=200, description="Clip title")
    layoutId: Optional[str] = Field(
        default=None,
        description="UUID of a saved layout from the layouts table.",
    )


class GenerateClipResponse(BaseModel):
    jobId: str
    clipId: str
    videoId: str
    status: str


class CreditsCostResponse(BaseModel):
    durationSeconds: int
    analysisCredits: int
    clipGenerationCredits: int


class CreditsCostByUrlRequest(BaseModel):
    url: HttpUrl


class CreditsCostByUrlResponse(BaseModel):
    valid_url: bool
    cost: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _raise_on_error(response, context: str):
    try:
        assert_response_ok(response, context)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


def _enqueue_or_fail(
    *,
    queue_name: str,
    task_path: str,
    job_data: dict,
    job_id: str,
    user_id: str,
    job_type: str,
    video_id: str,
    clip_id: str | None = None,
):
    try:
        enqueue_job(queue_name, task_path, job_data, job_id=job_id)
    except Exception as exc:
        err = f"Queue enqueue failed: {exc}"
        logger.error("Failed to enqueue job %s: %s", job_id, exc)

        fail_resp = (
            supabase.table("jobs")
            .update({"status": "failed", "error_message": err})
            .eq("id", job_id)
            .execute()
        )
        _raise_on_error(fail_resp, "Failed to mark job as failed after enqueue error")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to queue job right now. Please retry.",
        )

    logger.info(
        "Job %s enqueued on %s queue (type=%s user=%s video=%s clip=%s)",
        job_id,
        queue_name,
        job_type,
        user_id,
        video_id,
        clip_id,
    )


def _whisper_ready() -> bool:
    """Lazily verify Whisper is usable on this server process."""
    global _WHISPER_READY
    if _WHISPER_READY is not None:
        return _WHISPER_READY

    try:
        # Lazy import to avoid loading Whisper unless the cost endpoint needs it.
        from services.transcriber import Transcriber

        Transcriber()
        _WHISPER_READY = True
    except Exception as exc:
        logger.warning("Whisper readiness check failed: %s", exc)
        _WHISPER_READY = False

    return _WHISPER_READY


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/videos/analyze", response_model=AnalyzeVideoResponse)
def analyze_video(payload: AnalyzeVideoRequest) -> AnalyzeVideoResponse:
    job_id = str(uuid4())
    url_str = str(payload.url)

    # Reuse existing video for this user+url if present; otherwise use payload.videoId or new id
    existing = (
        supabase.table("videos")
        .select("id")
        .eq("url", url_str)
        .eq("user_id", payload.userId)
        .limit(1)
        .execute()
    )
    _raise_on_error(existing, "Failed to query existing video")
    if existing.data and len(existing.data) > 0:
        video_id = existing.data[0]["id"]
        logger.info("Reusing existing video %s for analyze url", video_id)
    else:
        video_id = payload.videoId or str(uuid4())

    logger.info(
        "Analyze request - user=%s  video=%s  url=%s  numClips=%d",
        payload.userId,
        video_id,
        url_str,
        payload.numClips,
    )

    # Ensure video + job rows exist so workers can update progress
    video_resp = (
        supabase.table("videos")
        .upsert(
            {
                "id": video_id,
                "user_id": payload.userId,
                "url": url_str,
                "status": "pending",
            },
            on_conflict="id",
        )
        .execute()
    )
    _raise_on_error(video_resp, "Failed to upsert video")

    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "userId": payload.userId,
        "url": url_str,
        "numClips": payload.numClips,
    }

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": payload.userId,
                "video_id": video_id,
                "type": "analyze_video",
                "status": "queued",
                "input_data": job_data,
            },
            on_conflict="id",
        )
        .execute()
    )
    _raise_on_error(job_resp, "Failed to upsert job")

    _enqueue_or_fail(
        queue_name="video-processing",
        task_path=ANALYZE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=payload.userId,
        job_type="analyze_video",
        video_id=video_id,
    )

    return AnalyzeVideoResponse(jobId=job_id, videoId=video_id, status="queued")


@app.post("/clips/generate", response_model=GenerateClipResponse)
def generate_clip(payload: GenerateClipRequest) -> GenerateClipResponse:
    job_id = str(uuid4())

    logger.info(
        "Generate clip request - user=%s  clip=%s",
        payload.userId,
        payload.clipId,
    )

    clip_resp = (
        supabase.table("clips")
        .select("id, video_id, user_id")
        .eq("id", payload.clipId)
        .execute()
    )
    _raise_on_error(clip_resp, "Failed to load clip")

    clip_rows = clip_resp.data or []
    clip = clip_rows[0] if clip_rows else None
    if clip is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip not found",
        )

    if clip["user_id"] != payload.userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clip does not belong to this user",
        )

    job_data = {
        "jobId": job_id,
        "clipId": payload.clipId,
        "userId": payload.userId,
        "layoutId": payload.layoutId,
    }

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": payload.userId,
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
    _raise_on_error(job_resp, "Failed to upsert job")

    _enqueue_or_fail(
        queue_name="clip-generation",
        task_path=GENERATE_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=payload.userId,
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


@app.post("/clips/custom", response_model=GenerateClipResponse)
def custom_clip(payload: CustomClipRequest) -> GenerateClipResponse:
    """Create a clip from a URL + start/end time in one step.

    Downloads the video, fetches transcript, and generates the clip
    all within a single worker task.
    """
    if payload.endTime <= payload.startTime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="endTime must be greater than startTime",
        )

    clip_id = str(uuid4())
    job_id = str(uuid4())
    url_str = str(payload.url)

    logger.info(
        "Custom clip request - user=%s  url=%s  %.2f–%.2f",
        payload.userId,
        url_str,
        payload.startTime,
        payload.endTime,
    )

    # Reuse existing video for this user+url if present; otherwise create one
    existing = (
        supabase.table("videos")
        .select("id")
        .eq("url", url_str)
        .eq("user_id", payload.userId)
        .limit(1)
        .execute()
    )
    _raise_on_error(existing, "Failed to query existing video")
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
                    "user_id": payload.userId,
                    "url": url_str,
                    "status": "pending",
                }
            )
            .execute()
        )
        _raise_on_error(video_resp, "Failed to insert video")

    # Create clip row
    clip_insert = {
        "id": clip_id,
        "video_id": video_id,
        "user_id": payload.userId,
        "start_time": payload.startTime,
        "end_time": payload.endTime,
        "title": payload.title,
        "status": "pending",
    }
    if payload.layoutId:
        clip_insert["layout_id"] = payload.layoutId
    clip_resp = (
        supabase.table("clips").insert(clip_insert).execute()
    )
    _raise_on_error(clip_resp, "Failed to insert clip")

    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "clipId": clip_id,
        "userId": payload.userId,
        "url": str(payload.url),
        "startTime": payload.startTime,
        "endTime": payload.endTime,
        "title": payload.title,
        "layoutId": payload.layoutId,
    }

    # Create job row
    job_resp = (
        supabase.table("jobs")
        .insert(
            {
                "id": job_id,
                "user_id": payload.userId,
                "video_id": video_id,
                "clip_id": clip_id,
                "type": "generate_clip",
                "status": "queued",
                "input_data": job_data,
            }
        )
        .execute()
    )
    _raise_on_error(job_resp, "Failed to insert job")

    _enqueue_or_fail(
        queue_name="clip-generation",
        task_path=CUSTOM_CLIP_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=payload.userId,
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

@app.post("/credits/cost", response_model=CreditsCostByUrlResponse)
def get_credit_cost_from_url(payload: CreditsCostByUrlRequest) -> CreditsCostByUrlResponse:
    """Validate URL for clipping and return total credit cost.

    A URL is valid when yt-dlp can resolve a downloadable source and captions
    are available either from the source metadata or through Whisper fallback.
    """
    url_str = str(payload.url)
    downloader = VideoDownloader()

    try:
        probe = downloader.probe_url(url_str)
    except Exception as exc:
        logger.warning("URL validation failed for cost endpoint (%s): %s", url_str, exc)
        return CreditsCostByUrlResponse(valid_url=False, cost=0)

    duration_seconds = int(probe.get("duration_seconds") or 0)
    can_download = bool(probe.get("can_download"))
    has_source_captions = bool(probe.get("has_captions"))
    has_audio = bool(probe.get("has_audio"))
    can_use_whisper = has_audio and _whisper_ready()
    has_captions = has_source_captions or can_use_whisper

    valid_url = can_download and has_captions and duration_seconds > 0
    if not valid_url:
        return CreditsCostByUrlResponse(valid_url=False, cost=0)

    analysis_cost = calculate_video_analysis_cost(duration_seconds)

    return CreditsCostByUrlResponse(valid_url=True, cost=analysis_cost)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)

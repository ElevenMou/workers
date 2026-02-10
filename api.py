import logging
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from config import CREDIT_COST_CLIP_GENERATION, calculate_video_analysis_cost
from utils.redis_client import clip_queue, video_queue
from utils.supabase_client import supabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.api")

app = FastAPI(title="Clipry Workers API", version="1.0.0")

# Use string paths so the API process does not eagerly import heavy worker deps
ANALYZE_TASK_PATH = "tasks.analyze_video.analyze_video_task"
GENERATE_TASK_PATH = "tasks.generate_clip.generate_clip_task"


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


class GenerateClipResponse(BaseModel):
    jobId: str
    clipId: str
    videoId: str
    status: str


class CreditsCostResponse(BaseModel):
    durationSeconds: int
    analysisCredits: int
    clipGenerationCredits: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _raise_on_error(response, context: str):
    error = getattr(response, "error", None)
    if error:
        detail = getattr(error, "message", None) or str(error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{context}: {detail}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/videos/analyze", response_model=AnalyzeVideoResponse)
def analyze_video(payload: AnalyzeVideoRequest) -> AnalyzeVideoResponse:
    video_id = payload.videoId or str(uuid4())
    job_id = str(uuid4())

    logger.info(
        "Analyze request - user=%s  video=%s  url=%s  numClips=%d",
        payload.userId,
        video_id,
        payload.url,
        payload.numClips,
    )

    # Ensure video + job rows exist so workers can update progress
    video_resp = (
        supabase.table("videos")
        .upsert(
            {
                "id": video_id,
                "user_id": payload.userId,
                "url": str(payload.url),
                "status": "pending",
            },
            on_conflict="id",
        )
        .execute()
    )
    _raise_on_error(video_resp, "Failed to upsert video")

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": payload.userId,
                "video_id": video_id,
                "type": "analyze_video",
                "status": "queued",
            },
            on_conflict="id",
        )
        .execute()
    )
    _raise_on_error(job_resp, "Failed to upsert job")

    job_data = {
        "jobId": job_id,
        "videoId": video_id,
        "userId": payload.userId,
        "url": str(payload.url),
        "numClips": payload.numClips,
    }

    video_queue.enqueue(ANALYZE_TASK_PATH, job_data, job_id=job_id)
    logger.info("Job %s enqueued on video-processing queue", job_id)

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

    job_resp = (
        supabase.table("jobs")
        .upsert(
            {
                "id": job_id,
                "user_id": payload.userId,
                "video_id": clip["video_id"],
                "type": "generate_clip",
                "status": "queued",
            },
            on_conflict="id",
        )
        .execute()
    )
    _raise_on_error(job_resp, "Failed to upsert job")

    job_data = {
        "jobId": job_id,
        "clipId": payload.clipId,
        "userId": payload.userId,
        "layoutId": payload.layoutId,
    }

    clip_queue.enqueue(GENERATE_TASK_PATH, job_data, job_id=job_id)
    logger.info("Job %s enqueued on clip-generation queue", job_id)

    return GenerateClipResponse(
        jobId=job_id,
        clipId=payload.clipId,
        videoId=clip["video_id"],
        status="queued",
    )


@app.get("/credits/cost", response_model=CreditsCostResponse)
def get_credit_cost(durationSeconds: int) -> CreditsCostResponse:
    if durationSeconds < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="durationSeconds must be non-negative",
        )

    return CreditsCostResponse(
        durationSeconds=durationSeconds,
        analysisCredits=calculate_video_analysis_cost(durationSeconds),
        clipGenerationCredits=CREDIT_COST_CLIP_GENERATION,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)

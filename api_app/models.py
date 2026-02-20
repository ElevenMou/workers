"""Pydantic request/response models for the workers API."""

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeVideoRequest(BaseModel):
    url: HttpUrl
    numClips: int = Field(default=5, ge=1, le=20)
    videoId: Optional[str] = Field(
        default=None,
        description="Optional existing video id to reuse",
    )


class AnalyzeVideoResponse(BaseModel):
    jobId: str
    videoId: str
    status: str


class GenerateClipRequest(BaseModel):
    clipId: str = Field(..., description="Clip id to generate")
    layoutId: Optional[str] = Field(
        default=None,
        description=(
            "UUID of a saved layout from the layouts table. "
            "The worker loads all generation settings (background, video, "
            "title, captions, quality) from this layout. "
            "When omitted, built-in defaults are used."
        ),
    )


class CustomClipRequest(BaseModel):
    url: HttpUrl = Field(..., description="Video URL (YouTube, etc.)")
    startTime: float = Field(..., ge=0, description="Clip start time in seconds")
    endTime: float = Field(..., gt=0, description="Clip end time in seconds")
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Clip title (duration must be 40-90 seconds)",
    )
    layoutId: Optional[str] = Field(
        default=None,
        description="UUID of a saved layout from the layouts table.",
    )


class GenerateClipResponse(BaseModel):
    jobId: str
    clipId: str
    videoId: str
    status: str


class ClipLayoutOptionsResponse(BaseModel):
    aspectRatios: list[str]
    recommendedAspectRatio: str
    videoScaleModes: list[str]


class CreditsCostByUrlRequest(BaseModel):
    url: HttpUrl


class CreditsCostByUrlResponse(BaseModel):
    valid_url: bool
    analysisCredits: int
    totalCredits: int


class CaptionModesResponse(BaseModel):
    modes: list[str]


class CaptionOptionsResponse(BaseModel):
    styles: list[str]
    animations: list[str]
    presets: list[dict]
    fontCases: list[str]
    positions: list[str]
    linesPerPageOptions: list[int]


class WorkerScaleRequest(BaseModel):
    videoWorkers: Optional[int] = Field(default=None, ge=0, le=64)
    clipWorkers: Optional[int] = Field(default=None, ge=0, le=64)


class WorkerScaleResponse(BaseModel):
    videoWorkers: int
    clipWorkers: int


class CaptionPresetsResponse(BaseModel):
    presets: list[dict]

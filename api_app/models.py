"""Pydantic request/response models for the workers API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl
from services.social.publish_config import PublishDestinationConfig


class AnalyzeVideoRequest(BaseModel):
    url: HttpUrl
    numClips: int = Field(default=5, ge=1, le=20)
    clipLengthSeconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=300,
        description=(
            "Legacy clip length selector (seconds). "
            "Used as fallback when clipLengthMinSeconds/clipLengthMaxSeconds are not provided."
        ),
    )
    clipLengthMinSeconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=300,
        description=(
            "Selected minimum clip length in seconds. Must be paired with clipLengthMaxSeconds."
        ),
    )
    clipLengthMaxSeconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=300,
        description=(
            "Selected maximum clip length in seconds. Must be paired with clipLengthMinSeconds."
        ),
    )
    processingStartSeconds: float = Field(
        default=0,
        ge=0,
        description="Optional analysis window start in seconds.",
    )
    processingEndSeconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional analysis window end in seconds.",
    )
    extraPrompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional additional guidance for the AI analyzer.",
    )
    videoId: Optional[str] = Field(
        default=None,
        description="Optional existing video id to reuse",
    )
    workspaceTeamId: Optional[str] = Field(
        default=None,
        description="Optional workspace team id hint (server resolves active workspace).",
    )


class AnalyzeVideoResponse(BaseModel):
    jobId: str
    videoId: str
    status: str


class BatchSplitVideoRequest(BaseModel):
    videoId: Optional[str] = Field(
        default=None,
        description="Optional existing video id to split into fixed-size parts.",
    )
    url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional direct source URL when splitting a new video.",
    )
    segmentLengthSeconds: Literal[60, 90] = Field(
        ...,
        description="Fixed output part length in seconds.",
    )
    layoutId: Optional[str] = Field(
        default=None,
        description="Optional template/layout id applied to every generated part.",
    )
    workspaceTeamId: Optional[str] = Field(
        default=None,
        description="Optional workspace team id hint (server resolves active workspace).",
    )


class BatchSplitVideoResponse(BaseModel):
    jobId: str
    videoId: str
    status: str


class GenerateClipRequest(BaseModel):
    clipId: str = Field(..., description="Clip id to generate")
    layoutId: Optional[str] = Field(
        default=None,
        description=(
            "Optional layout id to use for generation. When provided, "
            "this is authoritative if the layout belongs to the user."
        ),
    )
    smartCleanupEnabled: bool = Field(
        default=False,
        description=(
            "Whether to apply Smart Cleanup (remove filler words and long silences). "
            "Available for basic/pro/enterprise tiers only."
        ),
    )
    workspaceTeamId: Optional[str] = Field(
        default=None,
        description="Optional workspace team id hint (server resolves active workspace).",
    )


class CustomClipRequest(BaseModel):
    url: HttpUrl = Field(..., description="Video URL (YouTube, etc.)")
    startTime: float = Field(..., ge=0, description="Clip start time in seconds")
    endTime: float = Field(..., gt=0, description="Clip end time in seconds")
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Clip title",
    )
    layoutId: Optional[str] = Field(
        default=None,
        description=(
            "Optional layout id to use for generation. When provided, "
            "this is authoritative if the layout belongs to the user."
        ),
    )
    smartCleanupEnabled: bool = Field(
        default=False,
        description=(
            "Whether to apply Smart Cleanup (remove filler words and long silences). "
            "Available for basic/pro/enterprise tiers only."
        ),
    )
    workspaceTeamId: Optional[str] = Field(
        default=None,
        description="Optional workspace team id hint (server resolves active workspace).",
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
    numClips: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Estimated number of clips for generation cost projection.",
    )


class CreditsCostByUrlResponse(BaseModel):
    valid_url: bool
    analysisCredits: int
    totalCredits: int
    requestedClipCount: int = 0
    clipGenerationCreditsPerClip: int = 0
    estimatedGenerationCredits: int = 0
    smartCleanupSurchargePerClip: int = 0
    estimatedTotalCredits: int = 0
    analysisDurationSeconds: int | None = None
    maxAnalysisDurationSeconds: int | None = None
    durationLimitExceeded: bool = False
    currentBalance: Optional[int] = None
    hasEnoughCredits: Optional[bool] = None
    hasEnoughCreditsForEstimatedTotal: Optional[bool] = None
    videoTitle: Optional[str] = None
    thumbnailUrl: Optional[str] = None
    platform: Optional[str] = None
    detectedLanguage: Optional[str] = None


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
    socialWorkers: Optional[int] = Field(default=None, ge=0, le=64)


class WorkerScaleResponse(BaseModel):
    videoWorkers: int
    clipWorkers: int
    socialWorkers: int


class CaptionPresetsResponse(BaseModel):
    presets: list[dict]


class CreateClipPublicationsRequest(BaseModel):
    clipId: str = Field(..., description="Completed clip id to publish")
    destinations: list[PublishDestinationConfig] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "Validated per-destination publish configs for the active workspace."
        ),
    )
    clientRequestId: str = Field(..., min_length=1, max_length=255)


class ClipPublicationResponse(BaseModel):
    id: str
    batchId: str
    clipId: str
    socialAccountId: str
    provider: str
    status: str
    scheduledFor: str
    scheduledTimezone: str
    remotePostId: Optional[str] = None
    remotePostUrl: Optional[str] = None
    lastError: Optional[str] = None
    attemptCount: int = 0
    captionSnapshot: Optional[str] = None
    youtubeTitleSnapshot: Optional[str] = None
    resolvedConfig: Optional[PublishDestinationConfig] = None
    queuedAt: Optional[str] = None
    startedAt: Optional[str] = None
    publishedAt: Optional[str] = None
    failedAt: Optional[str] = None
    canceledAt: Optional[str] = None
    createdAt: Optional[str] = None
    accountDisplayName: Optional[str] = None


class CreateClipPublicationsResponse(BaseModel):
    batchId: str
    publications: list[ClipPublicationResponse]


class CancelClipPublicationResponse(BaseModel):
    publication: ClipPublicationResponse


class RetryClipPublicationResponse(BaseModel):
    publication: ClipPublicationResponse


class DeleteClipStorageRequest(BaseModel):
    storagePaths: list[str] = Field(default_factory=list, max_length=500)


class DeleteVideoStorageRequest(BaseModel):
    rawVideoPaths: list[str] = Field(default_factory=list, max_length=500)
    rawVideoStoragePaths: list[str] = Field(default_factory=list, max_length=500)


class DeleteStorageResponse(BaseModel):
    requested: int = 0
    removed: int = 0

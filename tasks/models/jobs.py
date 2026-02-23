"""Task job payload model types."""

from typing import NotRequired, TypedDict


class AnalyzeVideoJob(TypedDict):
    jobId: str
    videoId: str
    userId: str
    url: str
    numClips: NotRequired[int]
    analysisCredits: NotRequired[int]
    analysisDurationSeconds: NotRequired[int]
    workspaceTeamId: NotRequired[str | None]
    billingOwnerUserId: NotRequired[str | None]
    chargeSource: NotRequired[str]
    workspaceRole: NotRequired[str]


class GenerateClipJob(TypedDict):
    jobId: str
    clipId: str
    userId: str
    layoutId: NotRequired[str | None]
    smartCleanupEnabled: NotRequired[bool]
    generationCredits: NotRequired[int]
    clipRetentionDays: NotRequired[int | None]
    workspaceTeamId: NotRequired[str | None]
    billingOwnerUserId: NotRequired[str | None]
    chargeSource: NotRequired[str]
    workspaceRole: NotRequired[str]


class CustomClipJob(TypedDict):
    jobId: str
    videoId: str
    clipId: str
    userId: str
    url: str
    startTime: float
    endTime: float
    title: str
    layoutId: NotRequired[str | None]
    smartCleanupEnabled: NotRequired[bool]
    generationCredits: NotRequired[int]
    clipRetentionDays: NotRequired[int | None]
    workspaceTeamId: NotRequired[str | None]
    billingOwnerUserId: NotRequired[str | None]
    chargeSource: NotRequired[str]
    workspaceRole: NotRequired[str]

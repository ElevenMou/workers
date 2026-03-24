"""Task job payload model types."""

from typing import NotRequired, TypedDict


class AnalyzeVideoJob(TypedDict):
    jobId: str
    videoId: str
    userId: str
    url: str
    numClips: NotRequired[int]
    clipLengthSeconds: NotRequired[int]
    clipLengthMinSeconds: NotRequired[int]
    clipLengthMaxSeconds: NotRequired[int]
    processingStartSeconds: NotRequired[float]
    processingEndSeconds: NotRequired[float | None]
    extraPrompt: NotRequired[str | None]
    analysisCredits: NotRequired[int]
    analysisDurationSeconds: NotRequired[int]
    sourceTitle: NotRequired[str | None]
    sourceThumbnailUrl: NotRequired[str | None]
    sourcePlatform: NotRequired[str | None]
    sourceExternalId: NotRequired[str | None]
    sourceDetectedLanguage: NotRequired[str | None]
    sourceHasCaptions: NotRequired[bool | None]
    sourceHasAudio: NotRequired[bool | None]
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
    subscriptionTier: NotRequired[str]
    sourceMaxHeight: NotRequired[int | None]
    sourceProfile: NotRequired[str]
    masterProfile: NotRequired[str]
    deliveryProfile: NotRequired[str]
    outputQualityOverride: NotRequired[str | None]
    qualityPolicyProfile: NotRequired[str]


class SplitVideoJob(TypedDict):
    jobId: str
    videoId: str
    userId: str
    segmentLengthSeconds: int
    url: NotRequired[str | None]
    layoutId: NotRequired[str | None]
    expectedPartCount: NotRequired[int]
    expectedGenerationCredits: NotRequired[int]
    maxParts: NotRequired[int]
    clipRetentionDays: NotRequired[int | None]
    workspaceTeamId: NotRequired[str | None]
    billingOwnerUserId: NotRequired[str | None]
    chargeSource: NotRequired[str]
    workspaceRole: NotRequired[str]
    subscriptionTier: NotRequired[str]
    priorityProcessing: NotRequired[bool]
    sourceTitle: NotRequired[str | None]
    sourceThumbnailUrl: NotRequired[str | None]
    sourcePlatform: NotRequired[str | None]
    sourceExternalId: NotRequired[str | None]
    sourceDetectedLanguage: NotRequired[str | None]
    sourceHasCaptions: NotRequired[bool | None]
    sourceHasAudio: NotRequired[bool | None]
    sourceDurationSeconds: NotRequired[int]


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
    subscriptionTier: NotRequired[str]
    sourceMaxHeight: NotRequired[int | None]
    sourceProfile: NotRequired[str]
    masterProfile: NotRequired[str]
    deliveryProfile: NotRequired[str]
    outputQualityOverride: NotRequired[str | None]
    qualityPolicyProfile: NotRequired[str]


class PublishClipJob(TypedDict):
    jobId: str
    publicationId: str
    clipId: str
    userId: str
    workspaceTeamId: NotRequired[str | None]
    billingOwnerUserId: NotRequired[str | None]
    workspaceRole: NotRequired[str]
    subscriptionTier: NotRequired[str]
    provider: NotRequired[str]

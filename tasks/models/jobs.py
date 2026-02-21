"""Task job payload model types."""

from typing import NotRequired, TypedDict


class AnalyzeVideoJob(TypedDict):
    jobId: str
    videoId: str
    userId: str
    url: str
    numClips: NotRequired[int]
    analysisCredits: NotRequired[int]


class GenerateClipJob(TypedDict):
    jobId: str
    clipId: str
    userId: str
    layoutId: NotRequired[str]
    generationCredits: NotRequired[int]


class CustomClipJob(TypedDict):
    jobId: str
    videoId: str
    clipId: str
    userId: str
    url: str
    startTime: float
    endTime: float
    title: str
    layoutId: NotRequired[str]
    generationCredits: NotRequired[int]

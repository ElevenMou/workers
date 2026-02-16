"""Task job payload model types."""

from typing import NotRequired, TypedDict


class AnalyzeVideoJob(TypedDict):
    jobId: str
    videoId: str
    userId: str
    url: str
    numClips: NotRequired[int]


class GenerateClipJob(TypedDict):
    jobId: str
    clipId: str
    userId: str
    layoutId: NotRequired[str]


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

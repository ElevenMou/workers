"""Video-analysis task feature package.

Use lazy attribute loading so importing ``tasks.videos`` does not pull heavy
dependencies (e.g. Anthropic) during unrelated test collection.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "analyze_video_task",
    "cleanup_expired_raw_videos",
    "shift_transcript_timestamps",
    "transcript_has_word_timing",
    "transcribe_clip_window_with_whisper",
]


def __getattr__(name: str) -> Any:
    if name == "analyze_video_task":
        return import_module("tasks.videos.analyze").analyze_video_task
    if name == "cleanup_expired_raw_videos":
        return import_module("tasks.videos.cleanup").cleanup_expired_raw_videos
    if name in {
        "shift_transcript_timestamps",
        "transcript_has_word_timing",
        "transcribe_clip_window_with_whisper",
    }:
        transcript = import_module("tasks.videos.transcript")
        return getattr(transcript, name)
    raise AttributeError(f"module 'tasks.videos' has no attribute '{name}'")

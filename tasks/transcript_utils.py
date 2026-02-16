"""Compatibility wrapper for transcript helpers."""

from tasks.videos.transcript import (
    shift_transcript_timestamps,
    transcript_has_word_timing,
    transcribe_clip_window_with_whisper,
)

__all__ = [
    "shift_transcript_timestamps",
    "transcript_has_word_timing",
    "transcribe_clip_window_with_whisper",
]

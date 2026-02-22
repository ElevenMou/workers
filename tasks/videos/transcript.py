"""Transcript helper utilities shared by clip generation tasks."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import ffmpeg as ffmpeg_lib

from services.transcriber import Transcriber


def shift_transcript_timestamps(
    transcript: dict[str, Any],
    offset_seconds: float,
) -> dict[str, Any]:
    """Shift transcript segment/word timestamps by an absolute offset."""
    shifted = deepcopy(transcript or {})
    segments = shifted.get("segments")
    if not isinstance(segments, list):
        return shifted

    for seg in segments:
        if isinstance(seg.get("start"), (int, float)):
            seg["start"] = float(seg["start"]) + offset_seconds
        if isinstance(seg.get("end"), (int, float)):
            seg["end"] = float(seg["end"]) + offset_seconds

        words = seg.get("words")
        if isinstance(words, list):
            for word in words:
                if isinstance(word.get("start"), (int, float)):
                    word["start"] = float(word["start"]) + offset_seconds
                if isinstance(word.get("end"), (int, float)):
                    word["end"] = float(word["end"]) + offset_seconds

    return shifted


def transcript_has_word_timing(transcript: dict[str, Any] | None) -> bool:
    """Return whether the transcript includes per-word timing entries."""
    if not isinstance(transcript, dict):
        return False
    for seg in transcript.get("segments", []):
        words = seg.get("words")
        if isinstance(words, list) and words:
            return True
    return False


_WORD_LEVEL_STYLES = {"word_by_word", "karaoke"}


def needs_whisper_retranscription(
    transcript: dict[str, Any] | None,
    caption_style: str,
) -> bool:
    """Return True when the clip needs Whisper for accurate word timing.

    Conditions:
    1. The transcript source is "youtube" (no word-level timing).
    2. The caption style requires word-level timing (word_by_word or karaoke).
    """
    if caption_style.strip().lower() not in _WORD_LEVEL_STYLES:
        return False
    if not isinstance(transcript, dict):
        return False
    if transcript.get("source") != "youtube":
        return False
    if transcript_has_word_timing(transcript):
        return False
    return True


def transcribe_clip_window_with_whisper(
    *,
    media_path: str,
    work_dir: str,
    clip_id: str,
    start_time: float,
    end_time: float,
    video_duration_seconds: float,
) -> dict[str, Any]:
    """Transcribe only the clip window (+context) and return absolute timestamps."""
    context_pad = max(
        0.0,
        float(os.getenv("WHISPER_CLIP_CONTEXT_PAD_SECONDS", "1.5")),
    )
    window_start = max(0.0, start_time - context_pad)
    window_end = min(float(video_duration_seconds), end_time + context_pad)
    window_duration = max(1.0, window_end - window_start)

    clip_audio_path = os.path.join(work_dir, f"{clip_id}_window_audio.mp3")
    ffmpeg_lib.input(media_path, ss=window_start, t=window_duration).output(
        clip_audio_path,
        ac=1,
        ar=16000,
    ).overwrite_output().run(quiet=True)

    transcriber = Transcriber()
    transcript = transcriber.transcribe(clip_audio_path)
    transcript = shift_transcript_timestamps(transcript, window_start)
    transcript["source"] = "whisper"
    return transcript


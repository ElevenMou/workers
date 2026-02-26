"""Transcript helper utilities shared by clip generation tasks."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import ffmpeg as ffmpeg_lib

from config import WHISPER_CLIP_MODEL
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


def transcript_has_word_timing_in_window(
    transcript: dict[str, Any] | None,
    *,
    start_time: float,
    end_time: float,
    minimum_words: int = 1,
) -> bool:
    """Return whether transcript has usable word timing overlapping a window."""
    if not isinstance(transcript, dict):
        return False

    window_start = float(start_time)
    window_end = float(end_time)
    if window_end <= window_start:
        return False

    words_found = 0
    for seg in transcript.get("segments", []):
        words = seg.get("words")
        if not isinstance(words, list) or not words:
            continue

        for word in words:
            token = str(word.get("word", word.get("text", "")) or "").strip()
            if not token:
                continue
            try:
                word_start = float(word.get("start"))
                word_end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            if word_end <= word_start:
                continue
            if word_end <= window_start or word_start >= window_end:
                continue
            words_found += 1
            if words_found >= max(1, int(minimum_words)):
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
    model_name: str | None = None,
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

    selected_model = str(model_name or WHISPER_CLIP_MODEL).strip() or WHISPER_CLIP_MODEL
    transcriber = Transcriber(model_name=selected_model)
    transcript = transcriber.transcribe(clip_audio_path)
    transcript = shift_transcript_timestamps(transcript, window_start)
    transcript["source"] = "whisper"
    return transcript

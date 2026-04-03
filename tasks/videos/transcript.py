"""Transcript helper utilities shared by clip generation tasks."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any

import ffmpeg as ffmpeg_lib

from config import WHISPER_CLIP_MODEL
from services.transcriber import Transcriber
from tasks.clips.helpers.media import probe_has_audio_stream

_ANALYSIS_SEGMENT_WORD_CHUNK_SIZE = 8
_RTL_LANGUAGE_PREFIXES = ("ar", "fa", "he", "iw", "ps", "sd", "ug", "ur", "yi")
_RTL_SCRIPT_RE = re.compile(r"[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_language_code(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return text.replace("_", "-")


def transcript_language_code(transcript: dict[str, Any] | None) -> str | None:
    if not isinstance(transcript, dict):
        return None

    metadata = transcript.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for candidate in (
        transcript.get("languageCode"),
        transcript.get("language"),
        transcript.get("language_code"),
        metadata_dict.get("languageCode"),
        metadata_dict.get("language"),
        metadata_dict.get("language_code"),
    ):
        normalized = _normalize_language_code(candidate)
        if normalized:
            return normalized
    return None


def transcript_source_name(transcript: dict[str, Any] | None) -> str | None:
    if not isinstance(transcript, dict):
        return None

    metadata = transcript.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for candidate in (
        transcript.get("source"),
        transcript.get("transcript_source"),
        metadata_dict.get("source"),
        metadata_dict.get("transcript_source"),
    ):
        text = str(candidate or "").strip().lower()
        if text:
            return text
    return None


def _iter_transcript_text_tokens(transcript: dict[str, Any] | None) -> list[str]:
    if not isinstance(transcript, dict):
        return []

    tokens: list[str] = []
    for segment in transcript.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if text:
            tokens.append(text)
        words = segment.get("words")
        if isinstance(words, list):
            for word in words:
                token = str(word.get("word", word.get("text", "")) or "").strip()
                if token:
                    tokens.append(token)
        if len(tokens) >= 12:
            break
    return tokens


def transcript_is_rtl(transcript: dict[str, Any] | None) -> bool:
    language = transcript_language_code(transcript)
    if language:
        base = language.split("-", 1)[0]
        if base in _RTL_LANGUAGE_PREFIXES:
            return True

    return any(_RTL_SCRIPT_RE.search(token) for token in _iter_transcript_text_tokens(transcript))


def transcript_uses_non_english_source_captions(transcript: dict[str, Any] | None) -> bool:
    source = transcript_source_name(transcript)
    if not source or source == "whisper":
        return False

    language = transcript_language_code(transcript)
    if not language:
        return False

    return language.split("-", 1)[0] != "en"


def whisper_retranscription_skip_reason(transcript: dict[str, Any] | None) -> str | None:
    if transcript_is_rtl(transcript):
        return "rtl_transcript"
    if transcript_uses_non_english_source_captions(transcript):
        return "non_english_source_captions"
    return None


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


_WORD_LEVEL_STYLES = {"karaoke", "highlight", "highlight_box"}


def needs_whisper_retranscription(
    transcript: dict[str, Any] | None,
    caption_style: str,
) -> bool:
    """Return True when the clip needs Whisper for accurate word timing.

    Word-level caption styles need real per-word timing data. Even if the
    source transcript originally came from Whisper, long-form jobs may choose
    to store only segment-level timings for speed.
    """
    if caption_style.strip().lower() not in _WORD_LEVEL_STYLES:
        return False
    if not isinstance(transcript, dict):
        return True
    return not transcript_has_word_timing(transcript)


def _split_plain_segment_into_chunks(
    *,
    start: float,
    end: float,
    text: str,
    chunk_size: int = _ANALYSIS_SEGMENT_WORD_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    words = [word for word in str(text or "").split() if word]
    if not words:
        return []

    if len(words) <= chunk_size:
        return [{"start": start, "end": end, "text": " ".join(words)}]

    duration = max(0.0, end - start)
    if duration <= 0:
        return []

    time_per_word = duration / len(words)
    chunks: list[dict[str, Any]] = []
    for index in range(0, len(words), chunk_size):
        chunk_words = words[index : index + chunk_size]
        chunk_start = start + index * time_per_word
        chunk_end = start + min(index + chunk_size, len(words)) * time_per_word
        if chunk_end <= chunk_start:
            continue
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "text": " ".join(chunk_words),
            }
        )
    return chunks


def normalize_transcript_for_analysis_window(
    *,
    transcript: dict[str, Any],
    start_time: float,
    end_time: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clip, de-overlap, and normalize transcript segments for AI analysis."""
    source_segments = transcript.get("segments") or []
    stats = {
        "segments_total": len(source_segments),
        "segments_used": 0,
        "segments_dropped_partial_without_words": 0,
        "segments_dropped_invalid": 0,
        "segments_clipped_with_words": 0,
        "segments_split_plain_text": 0,
    }

    window_start = float(start_time)
    window_end = float(end_time)
    clipped_segments: list[dict[str, Any]] = []

    for segment in source_segments:
        seg_start = _as_float(segment.get("start"))
        seg_end = _as_float(segment.get("end"))
        text = str(segment.get("text") or "").strip()
        if seg_start is None or seg_end is None or seg_end <= seg_start or not text:
            stats["segments_dropped_invalid"] += 1
            continue

        if seg_end <= window_start or seg_start >= window_end:
            continue

        overlap_start = max(seg_start, window_start)
        overlap_end = min(seg_end, window_end)
        if overlap_end <= overlap_start:
            continue

        is_partial = overlap_start > seg_start or overlap_end < seg_end
        words = segment.get("words")
        if isinstance(words, list) and words:
            clipped_words: list[dict[str, Any]] = []
            for word in words:
                token = str(word.get("word", word.get("text", "")) or "").strip()
                word_start = _as_float(word.get("start"))
                word_end = _as_float(word.get("end"))
                if (
                    not token
                    or word_start is None
                    or word_end is None
                    or word_end <= word_start
                    or word_end <= overlap_start
                    or word_start >= overlap_end
                ):
                    continue

                clipped_start = max(word_start, overlap_start)
                clipped_end = min(word_end, overlap_end)
                if clipped_end <= clipped_start:
                    continue
                clipped_words.append(
                    {
                        "word": token,
                        "start": clipped_start,
                        "end": clipped_end,
                    }
                )

            if not clipped_words:
                continue

            clipped_words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
            clipped_segments.append(
                {
                    "start": float(clipped_words[0]["start"]),
                    "end": float(clipped_words[-1]["end"]),
                    "text": " ".join(str(word["word"]) for word in clipped_words).strip(),
                    "words": clipped_words,
                }
            )
            stats["segments_used"] += 1
            stats["segments_clipped_with_words"] += 1
            continue

        if is_partial:
            stats["segments_dropped_partial_without_words"] += 1
            continue

        clipped_segments.extend(
            _split_plain_segment_into_chunks(
                start=seg_start,
                end=seg_end,
                text=text,
            )
        )
        stats["segments_used"] += 1
        if len(text.split()) > _ANALYSIS_SEGMENT_WORD_CHUNK_SIZE:
            stats["segments_split_plain_text"] += 1

    clipped_segments.sort(key=lambda item: float(item["start"]))

    normalized_segments: list[dict[str, Any]] = []
    for segment in clipped_segments:
        segment_start = _as_float(segment.get("start"))
        segment_end = _as_float(segment.get("end"))
        text = str(segment.get("text") or "").strip()
        if segment_start is None or segment_end is None or segment_end <= segment_start or not text:
            continue

        if normalized_segments:
            previous = normalized_segments[-1]
            previous_end = float(previous["end"])
            if previous_end > segment_start:
                trimmed_end = max(float(previous["start"]), segment_start)
                previous["end"] = trimmed_end
                previous_words = previous.get("words")
                if isinstance(previous_words, list) and previous_words:
                    trimmed_words = []
                    for word in previous_words:
                        ws = max(float(word["start"]), float(previous["start"]))
                        we = min(float(word["end"]), float(previous["end"]))
                        if we <= ws:
                            continue
                        trimmed_words.append(
                            {
                                "word": word["word"],
                                "start": ws,
                                "end": we,
                            }
                        )
                    previous["words"] = trimmed_words
                    previous["text"] = " ".join(word["word"] for word in trimmed_words).strip()

                if previous["end"] <= previous["start"] or not str(previous.get("text") or "").strip():
                    normalized_segments.pop()

        if segment_end <= segment_start or not text:
            continue
        normalized_segments.append(segment)

    return normalized_segments, stats


def transcribe_clip_window_with_whisper(
    *,
    media_path: str,
    work_dir: str,
    clip_id: str,
    start_time: float,
    end_time: float,
    video_duration_seconds: float,
    model_name: str | None = None,
    language_hint: str | None = None,
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
    if not probe_has_audio_stream(media_path):
        raise RuntimeError(
            f"Source media has no usable audio stream for Whisper clip-window transcription: {media_path}"
        )
    try:
        ffmpeg_lib.input(media_path, ss=window_start, t=window_duration).output(
            clip_audio_path,
            ac=1,
            ar=16000,
        ).overwrite_output().run(quiet=True)
    except ffmpeg_lib.Error as exc:
        stderr = (getattr(exc, "stderr", None) or b"").decode("utf-8", errors="ignore").strip()
        detail = stderr.splitlines()[-1].strip() if stderr else str(exc)
        raise RuntimeError(
            f"Failed to extract clip-window audio for Whisper transcription: {detail}"
        ) from exc

    selected_model = str(model_name or WHISPER_CLIP_MODEL).strip() or WHISPER_CLIP_MODEL
    transcriber = Transcriber(model_name=selected_model)
    transcript = transcriber.transcribe(
        clip_audio_path,
        language_hint=language_hint,
        word_timestamps=True,
    )
    transcript = shift_transcript_timestamps(transcript, window_start)
    transcript["source"] = "whisper"
    return transcript

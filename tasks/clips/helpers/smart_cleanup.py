"""Smart Cleanup planning and rendering helpers.

This module provides deterministic timeline planning and transcript remapping
for clip-generation Smart Cleanup. The balanced profile removes common filler
tokens and compresses long silences.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from functools import lru_cache
from typing import Any, Iterable, TypedDict

import ffmpeg as ffmpeg_lib

try:
    import stopwordsiso
except ModuleNotFoundError:  # pragma: no cover - import-time guard for local envs
    stopwordsiso = None

BALANCED_SILENCE_CUTOFF_SECONDS = 0.55
BALANCED_RETAINED_SILENCE_SECONDS = 0.18
BALANCED_SILENCE_BOUNDARY_GUARD_SECONDS = 0.08
BALANCED_STOPWORD_PADDING_SECONDS = 0.02
BALANCED_FILLER_MAX_DURATION_SECONDS = 0.45
BALANCED_FILLER_MIN_EDGE_GAP_SECONDS = 0.10
BALANCED_STOPWORD_REPEAT_GAP_SECONDS = 0.24
BALANCED_STOPWORD_MAX_DURATION_SECONDS = 0.22
BALANCED_STOPWORD_MAX_CHARS = 5
_MIN_INTERVAL_SECONDS = 0.01
_SEGMENT_BREAK_GAP_SECONDS = 0.9
DEFAULT_SMART_CLEANUP_LANGUAGE = "en"
_LANGUAGE_FIELD_KEYS = (
    "language",
    "languageCode",
    "language_code",
    "lang",
    "locale",
)
_TOKEN_RE = re.compile(r"[^\w']+", flags=re.UNICODE)

_DEFAULT_FILLER_SINGLE_TOKENS = frozenset(
    {
        "um",
        "uh",
        "erm",
        "ah",
        "hmm",
        "mm",
        "mmm",
        "hm",
        "eh",
    }
)
_DEFAULT_FILLER_PHRASES = (
    ("you", "know"),
    ("i", "mean"),
    ("kind", "of"),
    ("sort", "of"),
)
_LANGUAGE_FILLER_SINGLE_TOKENS: dict[str, tuple[str, ...]] = {
    "en": ("like", "actually", "basically", "literally", "well"),
    "es": ("este", "pues", "bueno", "osea"),
    "fr": ("euh", "bah", "ben", "hein"),
    "de": ("ahm", "also"),
    "it": ("ehm", "cioe", "dunque"),
    "pt": ("tipo", "entao", "hum"),
}
_LANGUAGE_FILLER_PHRASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "es": (("o", "sea"),),
    "fr": (("tu", "sais"),),
}


class TimelineSegment(TypedDict):
    source_start: float
    source_end: float
    output_start: float
    output_end: float


class SmartCleanupSummary(TypedDict):
    stopwords_removed: int
    silence_seconds_removed: float
    original_duration_seconds: float
    output_duration_seconds: float


class SmartCleanupPlan(TypedDict):
    keep_intervals: list[tuple[float, float]]
    removal_intervals: list[tuple[float, float]]
    timeline_map: list[TimelineSegment]
    cleaned_transcript: dict[str, Any]
    summary: SmartCleanupSummary


class SmartCleanupResult(TypedDict):
    video_path: str
    transcript: dict[str, Any]
    keep_intervals: list[tuple[float, float]]
    summary: SmartCleanupSummary
    timeline_map: list[TimelineSegment]


def _normalize_language_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-")
    if not code:
        return None
    primary = code.split("-", 1)[0]
    if not primary or len(primary) < 2:
        return None
    return primary


def _resolve_transcript_languages(transcript: dict[str, Any] | None) -> tuple[str, ...]:
    resolved: list[str] = []
    if isinstance(transcript, dict):
        raw_candidates: list[Any] = [transcript.get(key) for key in _LANGUAGE_FIELD_KEYS]
        metadata = transcript.get("metadata")
        if isinstance(metadata, dict):
            raw_candidates.extend(metadata.get(key) for key in _LANGUAGE_FIELD_KEYS)

        for candidate in raw_candidates:
            language = _normalize_language_code(candidate)
            if language and language not in resolved:
                resolved.append(language)

    if DEFAULT_SMART_CLEANUP_LANGUAGE not in resolved:
        resolved.append(DEFAULT_SMART_CLEANUP_LANGUAGE)

    return tuple(resolved)


@lru_cache(maxsize=64)
def _load_multilingual_stopwords(languages: tuple[str, ...]) -> frozenset[str]:
    if stopwordsiso is None:
        return frozenset()

    loaded: set[str] = set()
    for language in languages:
        try:
            candidates = stopwordsiso.stopwords(language)
        except Exception:
            continue
        if isinstance(candidates, set):
            loaded.update(candidates)

    if not loaded and DEFAULT_SMART_CLEANUP_LANGUAGE not in languages:
        try:
            fallback = stopwordsiso.stopwords(DEFAULT_SMART_CLEANUP_LANGUAGE)
            if isinstance(fallback, set):
                loaded.update(fallback)
        except Exception:
            pass

    normalized: set[str] = set()
    for word in loaded:
        normalized_word = _normalize_token(str(word))
        if normalized_word:
            normalized.add(normalized_word)

    if not normalized:
        return frozenset()
    return frozenset(normalized)


@lru_cache(maxsize=64)
def _resolve_filler_lexicon(
    languages: tuple[str, ...],
) -> tuple[frozenset[str], tuple[tuple[str, ...], ...]]:
    single_tokens: set[str] = set(_DEFAULT_FILLER_SINGLE_TOKENS)
    phrase_tokens: set[tuple[str, ...]] = set(_DEFAULT_FILLER_PHRASES)

    for language in languages:
        for token in _LANGUAGE_FILLER_SINGLE_TOKENS.get(language, ()):
            normalized_token = _normalize_token(token)
            if normalized_token:
                single_tokens.add(normalized_token)

        for phrase in _LANGUAGE_FILLER_PHRASES.get(language, ()):
            normalized_phrase = tuple(
                token
                for token in (_normalize_token(part) for part in phrase)
                if token
            )
            if normalized_phrase:
                phrase_tokens.add(normalized_phrase)

    ordered_phrases = tuple(
        sorted(
            phrase_tokens,
            key=lambda value: (-len(value), value),
        )
    )
    return frozenset(single_tokens), ordered_phrases


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    lowered = (
        normalized.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("â€™", "'")
        .lower()
    )
    return _TOKEN_RE.sub("", lowered).replace("_", "").strip("'")


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    min_start: float,
    max_end: float,
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start, end in intervals:
        s = max(float(min_start), float(start))
        e = min(float(max_end), float(end))
        if e - s >= _MIN_INTERVAL_SECONDS:
            normalized.append((s, e))

    if not normalized:
        return []

    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = [normalized[0]]
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-6:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_keep_intervals(
    *,
    window_start: float,
    window_end: float,
    removal_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    cursor = float(window_start)

    for start, end in removal_intervals:
        if start > cursor + _MIN_INTERVAL_SECONDS:
            keep.append((cursor, start))
        cursor = max(cursor, end)

    if window_end > cursor + _MIN_INTERVAL_SECONDS:
        keep.append((cursor, float(window_end)))

    return keep


def build_timeline_map(keep_intervals: list[tuple[float, float]]) -> list[TimelineSegment]:
    timeline: list[TimelineSegment] = []
    output_cursor = 0.0

    for source_start, source_end in keep_intervals:
        duration = float(source_end) - float(source_start)
        if duration < _MIN_INTERVAL_SECONDS:
            continue
        timeline.append(
            {
                "source_start": float(source_start),
                "source_end": float(source_end),
                "output_start": output_cursor,
                "output_end": output_cursor + duration,
            }
        )
        output_cursor += duration

    return timeline


def map_source_time_to_output(
    *,
    source_time: float,
    timeline_map: list[TimelineSegment],
) -> float | None:
    target = float(source_time)
    for segment in timeline_map:
        seg_start = segment["source_start"]
        seg_end = segment["source_end"]
        if seg_start <= target <= seg_end:
            return segment["output_start"] + (target - seg_start)
    return None


def _sum_intervals(intervals: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in intervals:
        total += max(0.0, float(end) - float(start))
    return total


def _extract_words_in_window(
    *,
    transcript: dict[str, Any],
    window_start: float,
    window_end: float,
) -> list[dict[str, Any]]:
    raw_words: list[dict[str, Any]] = []
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not isinstance(segments, list):
        return raw_words

    for seg_index, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        seg_start = _as_float(seg.get("start"))
        seg_end = _as_float(seg.get("end"))
        if seg_start is None or seg_end is None:
            continue
        if seg_end <= window_start or seg_start >= window_end:
            continue

        seg_clamped_start = max(window_start, seg_start)
        seg_clamped_end = min(window_end, seg_end)
        if seg_clamped_end - seg_clamped_start < _MIN_INTERVAL_SECONDS:
            continue

        raw_seg_words = seg.get("words")
        if isinstance(raw_seg_words, list) and raw_seg_words:
            for word_index, item in enumerate(raw_seg_words):
                if not isinstance(item, dict):
                    continue
                token = str(item.get("word", item.get("text", ""))).strip()
                normalized = _normalize_token(token)
                if not normalized:
                    continue

                start_value = _as_float(item.get("start"))
                end_value = _as_float(item.get("end"))
                if start_value is None or end_value is None:
                    continue

                clamped_start = max(window_start, start_value)
                clamped_end = min(window_end, end_value)
                if clamped_end - clamped_start < _MIN_INTERVAL_SECONDS:
                    continue

                raw_words.append(
                    {
                        "word": token,
                        "normalized": normalized,
                        "start": clamped_start,
                        "end": clamped_end,
                        "segment_index": seg_index,
                        "word_index": word_index,
                    }
                )
            continue

        # Fallback for segment-level transcripts (for example, YouTube).
        text_tokens = [token for token in str(seg.get("text", "")).split() if token.strip()]
        if not text_tokens:
            continue

        seg_duration = seg_end - seg_start
        if seg_duration <= 0:
            continue
        per_word = seg_duration / len(text_tokens)
        for word_index, token in enumerate(text_tokens):
            normalized = _normalize_token(token)
            if not normalized:
                continue

            word_start = seg_start + (word_index * per_word)
            word_end = seg_start + ((word_index + 1) * per_word)
            clamped_start = max(window_start, word_start)
            clamped_end = min(window_end, word_end)
            if clamped_end - clamped_start < _MIN_INTERVAL_SECONDS:
                continue

            raw_words.append(
                {
                    "word": token,
                    "normalized": normalized,
                    "start": clamped_start,
                    "end": clamped_end,
                    "segment_index": seg_index,
                    "word_index": word_index,
                }
            )

    raw_words.sort(key=lambda item: (float(item["start"]), float(item["end"])))

    # De-overlap and enforce monotonic timings.
    normalized_words: list[dict[str, Any]] = []
    cursor = float(window_start)
    for word in raw_words:
        start = max(float(word["start"]), cursor)
        end = min(float(word["end"]), float(window_end))
        if end - start < _MIN_INTERVAL_SECONDS:
            continue
        normalized_words.append({**word, "start": start, "end": end})
        cursor = end

    return normalized_words


def _build_stopword_intervals(
    *,
    words: list[dict[str, Any]],
    window_start: float,
    window_end: float,
    padding_seconds: float,
    stopwords: frozenset[str],
    filler_tokens: frozenset[str],
    filler_phrases: tuple[tuple[str, ...], ...],
) -> tuple[list[tuple[float, float]], int]:
    stopword_intervals: list[tuple[float, float]] = []
    stopwords_removed = 0
    covered_indices: set[int] = set()

    for idx in range(len(words)):
        if idx in covered_indices:
            continue

        for phrase in filler_phrases:
            phrase_length = len(phrase)
            if phrase_length <= 1 or idx + phrase_length > len(words):
                continue

            phrase_indices = range(idx, idx + phrase_length)
            if any(pos in covered_indices for pos in phrase_indices):
                continue

            window = words[idx : idx + phrase_length]
            if tuple(str(item["normalized"]) for item in window) != phrase:
                continue

            prev_gap = (
                float("inf")
                if idx == 0
                else max(0.0, float(window[0]["start"]) - float(words[idx - 1]["end"]))
            )
            next_gap = (
                float("inf")
                if idx + phrase_length >= len(words)
                else max(
                    0.0,
                    float(words[idx + phrase_length]["start"]) - float(window[-1]["end"]),
                )
            )
            edge_pause = (
                prev_gap >= BALANCED_FILLER_MIN_EDGE_GAP_SECONDS
                or next_gap >= BALANCED_FILLER_MIN_EDGE_GAP_SECONDS
            )
            if not edge_pause:
                continue

            start = max(window_start, float(window[0]["start"]) - padding_seconds)
            end = min(window_end, float(window[-1]["end"]) + padding_seconds)
            if end - start < _MIN_INTERVAL_SECONDS:
                continue

            stopword_intervals.append((start, end))
            stopwords_removed += 1
            covered_indices.update(phrase_indices)
            break

    for idx, word in enumerate(words):
        if idx in covered_indices:
            continue

        normalized_word = str(word["normalized"])
        duration = float(word["end"]) - float(word["start"])
        if duration < _MIN_INTERVAL_SECONDS:
            continue

        prev_gap = (
            float("inf")
            if idx == 0
            else max(0.0, float(word["start"]) - float(words[idx - 1]["end"]))
        )
        next_gap = (
            float("inf")
            if idx + 1 >= len(words)
            else max(0.0, float(words[idx + 1]["start"]) - float(word["end"]))
        )
        repeated_forward = (
            idx + 1 < len(words)
            and str(words[idx + 1]["normalized"]) == normalized_word
            and next_gap <= BALANCED_STOPWORD_REPEAT_GAP_SECONDS
        )
        repeated_backward = (
            idx > 0
            and str(words[idx - 1]["normalized"]) == normalized_word
            and prev_gap <= BALANCED_STOPWORD_REPEAT_GAP_SECONDS
        )
        repeated = repeated_forward or repeated_backward
        edge_pause = (
            prev_gap >= BALANCED_FILLER_MIN_EDGE_GAP_SECONDS
            or next_gap >= BALANCED_FILLER_MIN_EDGE_GAP_SECONDS
        )

        should_remove = False
        if normalized_word in filler_tokens:
            should_remove = (
                duration <= BALANCED_FILLER_MAX_DURATION_SECONDS and (edge_pause or repeated)
            )
        elif normalized_word in stopwords:
            # Generic stopwords are only safe to trim when clearly repeated
            # ("the the", "de de"), which usually indicates stutter.
            should_remove = (
                repeated_forward
                and duration <= BALANCED_STOPWORD_MAX_DURATION_SECONDS
                and len(normalized_word) <= BALANCED_STOPWORD_MAX_CHARS
            )

        if not should_remove:
            continue

        start = max(window_start, float(word["start"]) - padding_seconds)
        end = min(window_end, float(word["end"]) + padding_seconds)
        if end - start < _MIN_INTERVAL_SECONDS:
            continue

        stopword_intervals.append((start, end))
        stopwords_removed += 1

    return stopword_intervals, stopwords_removed


def _build_silence_removal_intervals(
    *,
    words: list[dict[str, Any]],
    cutoff_seconds: float,
    retained_silence_seconds: float,
) -> list[tuple[float, float]]:
    if len(words) < 2:
        return []

    silence_intervals: list[tuple[float, float]] = []
    for current, nxt in zip(words, words[1:]):
        gap = float(nxt["start"]) - float(current["end"])
        if gap <= cutoff_seconds:
            continue

        target_gap = max(
            retained_silence_seconds,
            BALANCED_SILENCE_BOUNDARY_GUARD_SECONDS * 2.0,
        )
        remove_duration = gap - target_gap
        if remove_duration < _MIN_INTERVAL_SECONDS:
            continue

        # Keep a protected margin around spoken words to avoid clipping
        # consonants/vowels near uncertain word boundaries.
        start = float(current["end"]) + (target_gap / 2.0)
        end = float(nxt["start"]) - (target_gap / 2.0)
        if end - start >= _MIN_INTERVAL_SECONDS:
            silence_intervals.append((start, end))

    return silence_intervals


def _clip_word_to_timeline(
    *,
    word_start: float,
    word_end: float,
    timeline_map: list[TimelineSegment],
) -> tuple[float, float] | None:
    mapped_start: float | None = None
    mapped_end: float | None = None

    for segment in timeline_map:
        source_start = segment["source_start"]
        source_end = segment["source_end"]
        overlap_start = max(float(word_start), source_start)
        overlap_end = min(float(word_end), source_end)
        if overlap_end - overlap_start < _MIN_INTERVAL_SECONDS:
            continue

        out_start = segment["output_start"] + (overlap_start - source_start)
        out_end = segment["output_start"] + (overlap_end - source_start)
        if mapped_start is None:
            mapped_start = out_start
        mapped_end = out_end

    if mapped_start is None or mapped_end is None:
        return None

    if mapped_end - mapped_start < _MIN_INTERVAL_SECONDS:
        return None

    return (mapped_start, mapped_end)


def _build_cleaned_transcript(
    *,
    source_words: list[dict[str, Any]],
    timeline_map: list[TimelineSegment],
) -> dict[str, Any]:
    mapped_words: list[dict[str, Any]] = []
    for word in source_words:
        clipped = _clip_word_to_timeline(
            word_start=float(word["start"]),
            word_end=float(word["end"]),
            timeline_map=timeline_map,
        )
        if not clipped:
            continue
        mapped_words.append(
            {
                "word": str(word["word"]).strip(),
                "start": clipped[0],
                "end": clipped[1],
            }
        )

    mapped_words.sort(key=lambda item: (float(item["start"]), float(item["end"])))

    segments: list[dict[str, Any]] = []
    active_words: list[dict[str, Any]] = []
    for word in mapped_words:
        if not active_words:
            active_words = [word]
            continue

        previous_end = float(active_words[-1]["end"])
        next_start = float(word["start"])
        if next_start - previous_end > _SEGMENT_BREAK_GAP_SECONDS:
            segments.append(
                {
                    "start": float(active_words[0]["start"]),
                    "end": float(active_words[-1]["end"]),
                    "text": " ".join(item["word"] for item in active_words),
                    "words": active_words,
                }
            )
            active_words = [word]
        else:
            active_words.append(word)

    if active_words:
        segments.append(
            {
                "start": float(active_words[0]["start"]),
                "end": float(active_words[-1]["end"]),
                "text": " ".join(item["word"] for item in active_words),
                "words": active_words,
            }
        )

    full_text = " ".join(segment["text"] for segment in segments if segment.get("text")).strip()
    return {
        "source": "smart_cleanup",
        "text": full_text,
        "segments": segments,
    }


def plan_balanced_smart_cleanup(
    *,
    transcript: dict[str, Any] | None,
    start_time: float,
    end_time: float,
) -> SmartCleanupPlan:
    if not isinstance(transcript, dict):
        raise RuntimeError("Smart Cleanup requires a transcript with word-level timings.")

    window_start = float(start_time)
    window_end = float(end_time)
    if window_end <= window_start:
        raise RuntimeError("Invalid cleanup window: end time must be greater than start time.")

    words = _extract_words_in_window(
        transcript=transcript,
        window_start=window_start,
        window_end=window_end,
    )
    if not words:
        raise RuntimeError("Smart Cleanup could not find words in the requested clip window.")

    languages = _resolve_transcript_languages(transcript)
    stopwords = _load_multilingual_stopwords(languages)
    filler_tokens, filler_phrases = _resolve_filler_lexicon(languages)

    stopword_intervals, stopwords_removed = _build_stopword_intervals(
        words=words,
        window_start=window_start,
        window_end=window_end,
        padding_seconds=BALANCED_STOPWORD_PADDING_SECONDS,
        stopwords=stopwords,
        filler_tokens=filler_tokens,
        filler_phrases=filler_phrases,
    )
    silence_intervals = _build_silence_removal_intervals(
        words=words,
        cutoff_seconds=BALANCED_SILENCE_CUTOFF_SECONDS,
        retained_silence_seconds=BALANCED_RETAINED_SILENCE_SECONDS,
    )

    merged_silence_intervals = merge_intervals(
        silence_intervals,
        min_start=window_start,
        max_end=window_end,
    )
    removal_intervals = merge_intervals(
        [*stopword_intervals, *silence_intervals],
        min_start=window_start,
        max_end=window_end,
    )
    keep_intervals = build_keep_intervals(
        window_start=window_start,
        window_end=window_end,
        removal_intervals=removal_intervals,
    )
    if not keep_intervals:
        raise RuntimeError("Smart Cleanup removed the entire clip window.")

    timeline_map = build_timeline_map(keep_intervals)
    if not timeline_map:
        raise RuntimeError("Smart Cleanup failed to build timeline mapping.")

    output_duration = float(timeline_map[-1]["output_end"])
    cleaned_transcript = _build_cleaned_transcript(
        source_words=words,
        timeline_map=timeline_map,
    )
    if not cleaned_transcript.get("segments"):
        raise RuntimeError("Smart Cleanup produced an empty transcript after filtering.")

    summary: SmartCleanupSummary = {
        "stopwords_removed": int(stopwords_removed),
        "silence_seconds_removed": round(_sum_intervals(merged_silence_intervals), 3),
        "original_duration_seconds": round(window_end - window_start, 3),
        "output_duration_seconds": round(output_duration, 3),
    }

    return {
        "keep_intervals": keep_intervals,
        "removal_intervals": removal_intervals,
        "timeline_map": timeline_map,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
    }


def _probe_video_duration_seconds(path: str) -> float:
    probe = ffmpeg_lib.probe(path)
    duration = float(probe.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise RuntimeError(f"Failed to probe output duration for {path}.")
    return duration


def render_condensed_video_from_keep_intervals(
    *,
    input_video_path: str,
    keep_intervals: list[tuple[float, float]],
    output_path: str,
    work_dir: str,
    crf: int = 21,
    preset: str = "medium",
) -> float:
    if not keep_intervals:
        raise RuntimeError("Cannot render Smart Cleanup output without keep intervals.")

    segment_paths: list[str] = []
    for index, (start, end) in enumerate(keep_intervals):
        duration = float(end) - float(start)
        if duration < _MIN_INTERVAL_SECONDS:
            continue

        segment_path = os.path.join(work_dir, f"smart_cleanup_segment_{index:03d}.mp4")
        (
            ffmpeg_lib.input(input_video_path, ss=float(start), t=duration)
            .output(
                segment_path,
                vcodec="libx264",
                acodec="aac",
                crf=crf,
                preset=preset,
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
        segment_paths.append(segment_path)

    if not segment_paths:
        raise RuntimeError("Smart Cleanup produced no renderable keep intervals.")

    if len(segment_paths) == 1:
        shutil.copyfile(segment_paths[0], output_path)
        return _probe_video_duration_seconds(output_path)

    concat_list_path = os.path.join(work_dir, "smart_cleanup_concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as concat_file:
        for path in segment_paths:
            normalized_path = path.replace("\\", "/").replace("'", "'\\''")
            concat_file.write(f"file '{normalized_path}'\n")

    (
        ffmpeg_lib.input(concat_list_path, format="concat", safe=0)
        .output(
            output_path,
            vcodec="libx264",
            acodec="aac",
            crf=crf,
            preset=preset,
            movflags="+faststart",
        )
        .overwrite_output()
        .run(quiet=True, capture_stderr=True)
    )

    return _probe_video_duration_seconds(output_path)


def apply_balanced_smart_cleanup(
    *,
    transcript: dict[str, Any] | None,
    video_path: str,
    clip_id: str,
    work_dir: str,
    start_time: float,
    end_time: float,
) -> SmartCleanupResult:
    plan = plan_balanced_smart_cleanup(
        transcript=transcript,
        start_time=start_time,
        end_time=end_time,
    )

    output_path = os.path.join(work_dir, f"{clip_id}_smart_cleanup.mp4")
    rendered_duration = render_condensed_video_from_keep_intervals(
        input_video_path=video_path,
        keep_intervals=plan["keep_intervals"],
        output_path=output_path,
        work_dir=work_dir,
    )

    summary = dict(plan["summary"])
    summary["output_duration_seconds"] = round(float(rendered_duration), 3)

    return {
        "video_path": output_path,
        "transcript": plan["cleaned_transcript"],
        "keep_intervals": plan["keep_intervals"],
        "summary": summary,
        "timeline_map": plan["timeline_map"],
    }


__all__ = [
    "BALANCED_RETAINED_SILENCE_SECONDS",
    "BALANCED_SILENCE_CUTOFF_SECONDS",
    "BALANCED_STOPWORD_PADDING_SECONDS",
    "DEFAULT_SMART_CLEANUP_LANGUAGE",
    "apply_balanced_smart_cleanup",
    "build_keep_intervals",
    "build_timeline_map",
    "map_source_time_to_output",
    "merge_intervals",
    "plan_balanced_smart_cleanup",
    "render_condensed_video_from_keep_intervals",
]


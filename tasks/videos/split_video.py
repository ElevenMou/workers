"""Batch split a long source video into fixed-length short parts."""

from __future__ import annotations

import logging
import os
import re
import shutil
import traceback
from typing import Any
from uuid import uuid4

from config import (
    WHISPER_FULL_TRANSCRIPT_WORD_TIMESTAMPS,
    calculate_custom_clip_generation_cost,
)
from services.ai_analyzer import AIAnalyzer
from services.clips.quality_policy import resolve_clip_quality_policy
from services.clips.render_profiles import (
    DEFAULT_DELIVERY_PROFILE,
    DEFAULT_MASTER_PROFILE,
    DEFAULT_SOURCE_PROFILE,
)
from services.transcriber import Transcriber
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.lifecycle import build_progress_result_data
from tasks.clips.helpers.smart_cleanup import (
    build_keep_intervals,
    build_timeline_map,
    merge_intervals,
    render_condensed_video_from_keep_intervals,
)
from tasks.clips.helpers.source_video import (
    build_raw_video_metadata_update,
    resolve_source_video,
)
from tasks.models.jobs import GenerateClipJob, SplitVideoJob
from tasks.videos.source_transcript import resolve_source_transcript
from tasks.videos.transcript import normalize_transcript_for_analysis_window
from utils.media_storage import upload_raw_video
from utils.redis_client import enqueue_job
from utils.sentry_context import configure_job_scope
from utils.supabase_client import (
    assert_response_ok,
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
    update_job_status,
    update_video_status,
)
from utils.workdirs import create_work_dir

logger = logging.getLogger(__name__)

_GENERATE_TASK_PATH = "tasks.generate_clip.generate_clip_task"
_STANDARD_CLIP_QUEUE = "clip-generation"
_PRIORITY_CLIP_QUEUE = "clip-generation-priority"
_MAX_PARTS = 20
_MIN_INTERVAL_SECONDS = 0.01
_PART_TITLE_PREFIX_RE = re.compile(r"^\s*part\s+\d+\s*[-:]\s*", re.IGNORECASE)

_INTRO_PATTERNS = (
    re.compile(r"\b(welcome\s+back|welcome\s+to\s+my\s+channel)\b", re.IGNORECASE),
    re.compile(r"\b(hey|hi|hello)\s+(guys|everyone|folks|friends)\b", re.IGNORECASE),
    re.compile(r"\b(in\s+this\s+(video|episode)|today\s+(i|we)'?re?\s+going\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(before\s+we\s+get\s+started|quick\s+announcement)\b", re.IGNORECASE),
    re.compile(r"\b(let'?s\s+(get\s+into|dive\s+in|jump\s+in))\b", re.IGNORECASE),
)
_OUTRO_PATTERNS = (
    re.compile(r"\b(thanks?\s+for\s+watching|see\s+you\s+in\s+the\s+next)\b", re.IGNORECASE),
    re.compile(r"\b(that'?s\s+it\s+for\s+today|hope\s+you\s+enjoyed)\b", re.IGNORECASE),
    re.compile(r"\b(don'?t\s+forget\s+to\s+(like|subscribe|follow))\b", re.IGNORECASE),
)
_AD_PATTERNS = (
    re.compile(r"\b(this\s+video\s+is\s+sponsored\s+by|sponsored\s+by|sponsorship\s+from)\b", re.IGNORECASE),
    re.compile(r"\b(use\s+code\s+\w+|promo\s+code|affiliate\s+link)\b", re.IGNORECASE),
    re.compile(r"\b(our\s+partner|today'?s\s+sponsor|special\s+offer|limited\s+time\s+offer)\b", re.IGNORECASE),
    re.compile(r"\b(try\s+it\s+free|free\s+trial|discount|coupon)\b", re.IGNORECASE),
    re.compile(r"\b(link\s+in\s+(bio|description)|check\s+the\s+description)\b", re.IGNORECASE),
)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _clip_queue_name(priority_processing: bool) -> str:
    return _PRIORITY_CLIP_QUEUE if priority_processing else _STANDARD_CLIP_QUEUE


def _load_video_context_row(video_id: str) -> dict[str, Any]:
    response = (
        supabase.table("videos")
        .select(
            "id,status,transcript,raw_video_path,raw_video_storage_path,url,title,"
            "duration_seconds,thumbnail_url,platform,external_id"
        )
        .eq("id", video_id)
        .limit(1)
        .execute()
    )
    assert_response_ok(response, f"Failed to load source metadata for video {video_id}")
    data = response.data
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return data[0] if data else {}
    return {}


def _persist_video_raw_path(video_id: str, raw_video_path: str) -> None:
    payload = build_raw_video_metadata_update(raw_video_path)
    response = supabase.table("videos").update(payload).eq("id", video_id).execute()
    assert_response_ok(response, f"Failed to persist raw video path for {video_id}")


def _persist_video_raw_metadata(video_id: str, payload: dict[str, Any]) -> None:
    response = supabase.table("videos").update(dict(payload)).eq("id", video_id).execute()
    assert_response_ok(response, f"Failed to persist raw video metadata for {video_id}")


def _update_split_job_progress(
    job_id: str,
    progress: int,
    stage: str,
    *,
    detail_key: str | None = None,
    detail_params: dict[str, Any] | None = None,
    extra_result_data: dict[str, Any] | None = None,
) -> None:
    update_job_status(
        job_id,
        "processing",
        progress,
        result_data=build_progress_result_data(
            stage=stage,
            detail_key=detail_key,
            detail_params=detail_params,
            extra_result_data=extra_result_data,
        ),
    )


def _best_effort_mark_failed(
    *,
    job_id: str,
    video_id: str,
    error_message: str,
    manage_video_status: bool,
) -> None:
    try:
        update_job_status(job_id, "failed", 0, error_message)
    except Exception as exc:
        logger.warning("[%s] Failed to mark split job failed: %s", job_id, exc)
    if manage_video_status:
        try:
            update_video_status(video_id, "failed", error_message=error_message)
        except Exception as exc:
            logger.warning("[%s] Failed to mark video %s failed: %s", job_id, video_id, exc)


def _is_latest_split_job_for_video(*, job_id: str, video_id: str) -> bool:
    response = (
        supabase.table("jobs")
        .select("id")
        .eq("video_id", video_id)
        .eq("type", "split_video")
        .order("created_at", desc=True)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    assert_response_ok(response, f"Failed to load latest split job for video {video_id}")
    rows = response.data or []
    if not rows:
        return True
    return str(rows[0].get("id") or "").strip() == job_id


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _find_heuristic_removal_ranges(
    transcript: dict[str, Any] | None,
    *,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(transcript, dict):
        return []

    intro_window = max(45.0, duration_seconds * 0.12)
    outro_window = max(45.0, duration_seconds * 0.12)
    raw_ranges: list[dict[str, Any]] = []

    for segment in transcript.get("segments", []) or []:
        start = _as_float(segment.get("start"))
        end = _as_float(segment.get("end"))
        text = _normalize_text(segment.get("text"))
        if start is None or end is None or end <= start or not text:
            continue

        kind: str | None = None
        confidence = 0.0
        reason: str | None = None
        if any(pattern.search(text) for pattern in _AD_PATTERNS):
            kind = "ad"
            confidence = 0.92
            reason = "Matched sponsor or promotional language"
        elif start <= intro_window and any(pattern.search(text) for pattern in _INTRO_PATTERNS):
            kind = "intro"
            confidence = 0.8
            reason = "Matched likely intro/setup language near the beginning"
        elif end >= max(0.0, duration_seconds - outro_window) and any(
            pattern.search(text) for pattern in _OUTRO_PATTERNS
        ):
            kind = "outro"
            confidence = 0.82
            reason = "Matched likely wrap-up or CTA language near the end"

        if kind is None:
            continue

        raw_ranges.append(
            {
                "kind": kind,
                "start": float(start),
                "end": float(end),
                "confidence": confidence,
                "reason": reason,
                "source": "heuristic",
            }
        )

    if not raw_ranges:
        return []

    merged: list[dict[str, Any]] = []
    for candidate in raw_ranges:
        if not merged:
            merged.append(dict(candidate))
            continue

        previous = merged[-1]
        same_kind = previous["kind"] == candidate["kind"]
        nearby = float(candidate["start"]) <= float(previous["end"]) + 2.5
        if same_kind and nearby:
            previous["end"] = max(float(previous["end"]), float(candidate["end"]))
            previous["confidence"] = max(
                float(previous["confidence"]),
                float(candidate["confidence"]),
            )
            continue

        merged.append(dict(candidate))

    return merged


def _combine_removal_ranges(
    *,
    heuristic_ranges: list[dict[str, Any]],
    ai_ranges: list[dict[str, Any]],
    duration_seconds: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    normalized_candidates: list[dict[str, Any]] = []
    for candidate in [*heuristic_ranges, *ai_ranges]:
        start = max(0.0, float(candidate.get("start") or 0.0))
        end = min(float(duration_seconds), float(candidate.get("end") or 0.0))
        if end - start < _MIN_INTERVAL_SECONDS:
            continue
        normalized_candidates.append(
            {
                "kind": str(candidate.get("kind") or "").strip().lower() or "ad",
                "start": start,
                "end": end,
                "confidence": max(0.0, min(1.0, float(candidate.get("confidence") or 0.0))),
                "reason": candidate.get("reason"),
                "source": str(candidate.get("source") or "heuristic").strip().lower() or "heuristic",
            }
        )

    if not normalized_candidates:
        return [], {
            "heuristic_ranges": heuristic_ranges,
            "ai_ranges": ai_ranges,
            "accepted_ranges": [],
            "final_removal_intervals": [],
        }

    normalized_candidates.sort(key=lambda item: (float(item["start"]), float(item["end"])))

    merged_candidates: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        if not merged_candidates:
            merged_candidates.append(
                {
                    "kind": candidate["kind"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "confidence": candidate["confidence"],
                    "sources": {candidate["source"]},
                    "reasons": [candidate.get("reason")] if candidate.get("reason") else [],
                }
            )
            continue

        previous = merged_candidates[-1]
        if float(candidate["start"]) <= float(previous["end"]) + 1.0:
            previous["end"] = max(float(previous["end"]), float(candidate["end"]))
            previous["confidence"] = max(float(previous["confidence"]), float(candidate["confidence"]))
            previous["sources"].add(candidate["source"])
            if candidate.get("reason"):
                previous["reasons"].append(candidate["reason"])
            if float(candidate["confidence"]) >= float(previous["confidence"]):
                previous["kind"] = candidate["kind"]
            continue

        merged_candidates.append(
            {
                "kind": candidate["kind"],
                "start": candidate["start"],
                "end": candidate["end"],
                "confidence": candidate["confidence"],
                "sources": {candidate["source"]},
                "reasons": [candidate.get("reason")] if candidate.get("reason") else [],
            }
        )

    accepted_ranges: list[dict[str, Any]] = []
    for candidate in merged_candidates:
        confidence = float(candidate["confidence"])
        if len(candidate["sources"]) > 1:
            confidence = min(0.98, confidence + 0.1)
        if confidence < 0.75:
            continue
        accepted_ranges.append(
            {
                "kind": candidate["kind"],
                "start": round(float(candidate["start"]), 3),
                "end": round(float(candidate["end"]), 3),
                "confidence": round(confidence, 3),
                "sources": sorted(candidate["sources"]),
                "reasons": [reason for reason in candidate["reasons"] if reason],
            }
        )

    merged_intervals = merge_intervals(
        [
            (float(candidate["start"]), float(candidate["end"]))
            for candidate in accepted_ranges
        ],
        min_start=0.0,
        max_end=float(duration_seconds),
    )

    total_removed = sum(end - start for start, end in merged_intervals)
    if duration_seconds > 0 and total_removed > duration_seconds * 0.55:
        logger.warning(
            "Batch split removal looked too aggressive (removed %.2fs of %.2fs); falling back to original source.",
            total_removed,
            duration_seconds,
        )
        merged_intervals = []
        accepted_ranges = []

    diagnostics = {
        "heuristic_ranges": heuristic_ranges,
        "ai_ranges": ai_ranges,
        "accepted_ranges": accepted_ranges,
        "final_removal_intervals": [[round(start, 3), round(end, 3)] for start, end in merged_intervals],
        "total_removed_seconds": round(total_removed, 3),
    }
    return merged_intervals, diagnostics


def _map_interval_to_output(
    *,
    start: float,
    end: float,
    timeline_map: list[dict[str, float]],
) -> tuple[float, float] | None:
    for segment in timeline_map:
        source_start = float(segment["source_start"])
        source_end = float(segment["source_end"])
        if start < source_start - 1e-6 or end > source_end + 1e-6:
            continue
        mapped_start = float(segment["output_start"]) + (float(start) - source_start)
        mapped_end = float(segment["output_start"]) + (float(end) - source_start)
        if mapped_end - mapped_start < _MIN_INTERVAL_SECONDS:
            return None
        return (mapped_start, mapped_end)
    return None


def _remap_transcript_to_clean_timeline(
    *,
    transcript: dict[str, Any] | None,
    timeline_map: list[dict[str, float]],
) -> dict[str, Any]:
    if not isinstance(transcript, dict):
        return {"segments": [], "text": "", "source": "batch_split_cleanup"}

    mapped_segments: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []) or []:
        seg_start = _as_float(segment.get("start"))
        seg_end = _as_float(segment.get("end"))
        text = _normalize_text(segment.get("text"))
        if seg_start is None or seg_end is None or seg_end <= seg_start or not text:
            continue

        words = segment.get("words")
        if isinstance(words, list) and words:
            mapped_words: list[dict[str, Any]] = []
            for word in words:
                word_text = _normalize_text(word.get("word", word.get("text")))
                word_start = _as_float(word.get("start"))
                word_end = _as_float(word.get("end"))
                if word_start is None or word_end is None or word_end <= word_start or not word_text:
                    continue
                mapped = _map_interval_to_output(
                    start=word_start,
                    end=word_end,
                    timeline_map=timeline_map,
                )
                if mapped is None:
                    continue
                mapped_words.append(
                    {
                        "word": word_text,
                        "start": mapped[0],
                        "end": mapped[1],
                    }
                )

            if mapped_words:
                mapped_segments.append(
                    {
                        "start": float(mapped_words[0]["start"]),
                        "end": float(mapped_words[-1]["end"]),
                        "text": " ".join(word["word"] for word in mapped_words).strip(),
                        "words": mapped_words,
                    }
                )
            continue

        mapped = _map_interval_to_output(
            start=seg_start,
            end=seg_end,
            timeline_map=timeline_map,
        )
        if mapped is None:
            continue
        mapped_segments.append(
            {
                "start": mapped[0],
                "end": mapped[1],
                "text": text,
            }
        )

    mapped_segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    remapped = dict(transcript)
    remapped["segments"] = mapped_segments
    remapped["text"] = " ".join(
        str(segment.get("text") or "").strip()
        for segment in mapped_segments
        if str(segment.get("text") or "").strip()
    ).strip()
    remapped["source"] = "batch_split_cleanup"
    return remapped


def _slice_transcript_for_window(
    *,
    transcript: dict[str, Any] | None,
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    if not isinstance(transcript, dict):
        return {"segments": [], "text": "", "source": "batch_split_cleanup"}

    segments, _stats = normalize_transcript_for_analysis_window(
        transcript=transcript,
        start_time=float(start_time),
        end_time=float(end_time),
    )
    clipped = dict(transcript)
    clipped["segments"] = segments
    clipped["text"] = " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if str(segment.get("text") or "").strip()
    ).strip()
    return clipped


def _build_part_windows(
    *,
    duration_seconds: float,
    segment_length_seconds: int,
    max_parts: int = _MAX_PARTS,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    cursor = 0.0
    index = 1
    safe_duration = max(0.0, float(duration_seconds))
    segment_length = max(1, int(segment_length_seconds))

    while cursor < safe_duration - _MIN_INTERVAL_SECONDS and index <= max_parts:
        end = min(safe_duration, cursor + float(segment_length))
        if end - cursor < _MIN_INTERVAL_SECONDS:
            break
        windows.append(
            {
                "index": index,
                "start": round(cursor, 6),
                "end": round(end, 6),
            }
        )
        cursor = end
        index += 1

    return windows


def _title_prompt_text(transcript: dict[str, Any] | None, *, max_chars: int = 320) -> str:
    if not isinstance(transcript, dict):
        return ""
    text = _normalize_text(transcript.get("text"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _fallback_part_title(index: int, source_title: str | None) -> str:
    cleaned_source_title = _normalize_text(source_title)
    if cleaned_source_title:
        return f"Part {index} - {cleaned_source_title}"
    return f"Part {index}"


def _format_part_title(index: int, title_suffix: str | None, source_title: str | None) -> str:
    cleaned_suffix = _PART_TITLE_PREFIX_RE.sub("", _normalize_text(title_suffix))
    if not cleaned_suffix:
        return _fallback_part_title(index, source_title)
    return f"Part {index} - {cleaned_suffix}"


def _enqueue_generate_child_job(
    *,
    job_data: GenerateClipJob,
    queue_name: str,
    clip_id: str,
    video_id: str,
    user_id: str,
    team_id: str | None,
    billing_owner_user_id: str,
) -> tuple[str, bool]:
    job_id = str(job_data["jobId"])
    response = (
        supabase.table("jobs")
        .insert(
            {
                "id": job_id,
                "user_id": user_id,
                "team_id": team_id,
                "billing_owner_user_id": billing_owner_user_id,
                "video_id": video_id,
                "clip_id": clip_id,
                "type": "generate_clip",
                "status": "queued",
                "input_data": job_data,
            }
        )
        .execute()
    )
    assert_response_ok(response, f"Failed to insert child generation job {job_id}")

    try:
        enqueue_job(queue_name, _GENERATE_TASK_PATH, job_data, job_id=job_id)
    except Exception as exc:
        logger.error("[%s] Failed to enqueue child generation job for clip %s: %s", job_id, clip_id, exc)
        try:
            update_job_status(job_id, "failed", 0, f"Queue enqueue failed: {exc}")
        except Exception:
            logger.warning("[%s] Failed to mark child generation job failed", job_id)
        try:
            supabase.table("clips").update(
                {
                    "status": "failed",
                    "error_message": f"Clip queue enqueue failed: {exc}",
                }
            ).eq("id", clip_id).execute()
        except Exception:
            logger.warning("[%s] Failed to mark clip %s failed after enqueue error", job_id, clip_id)
        return job_id, False

    logger.info(
        "Child generate job %s enqueued on %s queue (clip=%s video=%s user=%s)",
        job_id,
        queue_name,
        clip_id,
        video_id,
        user_id,
    )
    return job_id, True


def split_video_task(job_data: SplitVideoJob) -> None:
    job_id = str(job_data["jobId"])
    video_id = str(job_data["videoId"])
    user_id = str(job_data["userId"])
    workspace_team_id = job_data.get("workspaceTeamId")
    billing_owner_user_id = str(job_data.get("billingOwnerUserId") or user_id)
    charge_source = str(job_data.get("chargeSource") or "owner_wallet")
    workspace_role = str(job_data.get("workspaceRole") or "owner")
    subscription_tier = str(job_data.get("subscriptionTier") or "basic")
    segment_length_seconds = int(job_data.get("segmentLengthSeconds") or 60)
    layout_id = str(job_data.get("layoutId") or "").strip() or None
    expected_part_count = max(1, int(job_data.get("expectedPartCount") or 1))
    expected_generation_credits = max(0, int(job_data.get("expectedGenerationCredits") or 0))
    clip_retention_days = job_data.get("clipRetentionDays")
    priority_processing = bool(job_data.get("priorityProcessing"))
    source_title = str(job_data.get("sourceTitle") or "").strip() or None
    source_thumbnail_url = str(job_data.get("sourceThumbnailUrl") or "").strip() or None
    source_platform = str(job_data.get("sourcePlatform") or "").strip().lower() or None
    source_external_id = str(job_data.get("sourceExternalId") or "").strip() or None
    source_detected_language = str(job_data.get("sourceDetectedLanguage") or "").strip() or None
    source_has_audio = _as_optional_bool(job_data.get("sourceHasAudio"))
    source_url = str(job_data.get("url") or "").strip() or None

    configure_job_scope(
        job_id=job_id,
        job_type="split_video",
        user_id=user_id,
        video_id=video_id,
    )

    work_dir: str | None = None
    downloader: VideoDownloader | None = None
    analyzer: AIAnalyzer | None = None
    manage_video_status = False

    try:
        existing_video = _load_video_context_row(video_id)
        current_status = str(existing_video.get("status") or "").strip().lower()
        manage_video_status = current_status != "completed"

        if not _is_latest_split_job_for_video(job_id=job_id, video_id=video_id):
            update_job_status(
                job_id,
                "failed",
                0,
                "Superseded by a newer split-video request for the same source.",
                result_data={"stage": "superseded", "video_id": video_id},
            )
            return

        if expected_generation_credits > 0 and not has_sufficient_credits(
            user_id=billing_owner_user_id,
            amount=expected_generation_credits,
            charge_source=charge_source,
            team_id=workspace_team_id,
        ):
            available = (
                get_team_wallet_balance(workspace_team_id)
                if charge_source == "team_wallet" and workspace_team_id
                else get_credit_balance(billing_owner_user_id)
            )
            raise RuntimeError(
                "Insufficient credits for split-video generation before processing starts: "
                f"required={expected_generation_credits}, available={available}"
            )

        work_dir = create_work_dir(f"split_{job_id}")
        downloader = VideoDownloader(work_dir=work_dir)
        analyzer = AIAnalyzer()

        source_url = source_url or str(existing_video.get("url") or "").strip() or None
        if not source_url:
            raise RuntimeError("Split-video job is missing a usable source URL")

        if manage_video_status:
            update_video_status(video_id, "downloading")
        _update_split_job_progress(job_id, 5, "starting", detail_key="preparing_batch_split")

        source_resolution = resolve_source_video(
            video_id=video_id,
            source_url=source_url,
            initial_raw_video_path=existing_video.get("raw_video_path"),
            initial_raw_video_storage_path=existing_video.get("raw_video_storage_path"),
            downloader=downloader,
            load_video_row=_load_video_context_row,
            persist_raw_video_path=_persist_video_raw_path,
            persist_raw_video_metadata=_persist_video_raw_metadata,
            logger=logger,
            job_id=job_id,
            source_profile=DEFAULT_SOURCE_PROFILE,
        )
        source_video_path = source_resolution.video_path
        source_storage_path = (
            str(source_resolution.storage_path or "").strip()
            or str(existing_video.get("raw_video_storage_path") or "").strip()
            or None
        )

        source_info = {
            "title": source_title or existing_video.get("title"),
            "thumbnail": source_thumbnail_url or existing_video.get("thumbnail_url"),
            "platform": source_platform or existing_video.get("platform"),
            "external_id": source_external_id or existing_video.get("external_id"),
            "duration": int(job_data.get("sourceDurationSeconds") or existing_video.get("duration_seconds") or 0),
        }
        if source_resolution.download_metadata:
            source_info["duration"] = int(
                source_info["duration"] or source_resolution.download_metadata.get("duration") or 0
            )
            source_info["title"] = source_info["title"] or source_resolution.download_metadata.get("title")
            source_info["thumbnail"] = source_info["thumbnail"] or source_resolution.download_metadata.get("thumbnail")
            source_info["platform"] = source_info["platform"] or source_resolution.download_metadata.get("platform")
            source_info["external_id"] = source_info["external_id"] or source_resolution.download_metadata.get("external_id")

        if not source_info["duration"] or not source_info["platform"]:
            _update_split_job_progress(job_id, 12, "resolving_source", detail_key="checking_source_metadata")
            probe_data = downloader.probe_url(source_url)
            source_info["duration"] = int(source_info["duration"] or probe_data.get("duration_seconds") or 0)
            source_info["title"] = source_info["title"] or probe_data.get("title")
            source_info["thumbnail"] = source_info["thumbnail"] or probe_data.get("thumbnail")
            source_info["platform"] = source_info["platform"] or probe_data.get("platform")
            source_info["external_id"] = source_info["external_id"] or probe_data.get("external_id")
            if not source_detected_language:
                source_detected_language = str(probe_data.get("detected_language") or "").strip() or None
            if source_has_audio is None:
                source_has_audio = _as_optional_bool(probe_data.get("has_audio"))

        duration_seconds = int(source_info.get("duration") or 0)
        if duration_seconds <= 0:
            raise RuntimeError("Could not determine source video duration for batch split")

        def _transcribe_with_whisper(language_hint: str | None) -> tuple[dict[str, Any], bool]:
            if source_has_audio is False:
                raise RuntimeError("Source has no usable audio for transcript fallback")
            audio_path = downloader.extract_audio(source_video_path)
            transcript_payload = Transcriber().transcribe(
                audio_path,
                language_hint=language_hint,
                word_timestamps=WHISPER_FULL_TRANSCRIPT_WORD_TIMESTAMPS,
            )
            transcript_payload["source"] = "whisper"
            return transcript_payload, True

        if manage_video_status:
            update_video_status(video_id, "analyzing")
        _update_split_job_progress(job_id, 20, "fetching_transcript", detail_key="fetching_source_captions")

        transcript_resolution = resolve_source_transcript(
            existing_transcript=existing_video.get("transcript"),
            downloader=downloader,
            source_url=source_url,
            source_platform=str(source_info.get("platform") or "").strip().lower() or None,
            source_external_id=str(source_info.get("external_id") or "").strip() or None,
            source_detected_language=source_detected_language,
            source_has_audio=source_has_audio,
            whisper_fallback=_transcribe_with_whisper,
            job_id=job_id,
        )
        transcript = transcript_resolution.transcript

        video_update_payload = {
            "title": source_info.get("title"),
            "thumbnail_url": source_info.get("thumbnail"),
            "platform": source_info.get("platform"),
            "external_id": source_info.get("external_id"),
            "duration_seconds": duration_seconds,
            "transcript": transcript,
        }
        if source_storage_path:
            video_update_payload["raw_video_storage_path"] = source_storage_path
        video_metadata_response = (
            supabase.table("videos")
            .update(video_update_payload)
            .eq("id", video_id)
            .execute()
        )
        assert_response_ok(video_metadata_response, f"Failed to update video metadata for {video_id}")

        _update_split_job_progress(job_id, 32, "planning_cleanup", detail_key="analyzing_intro_outro_ad_ranges")

        heuristic_ranges = _find_heuristic_removal_ranges(
            transcript,
            duration_seconds=float(duration_seconds),
        )
        ai_ranges: list[dict[str, Any]] = []
        try:
            ai_ranges = analyzer.find_removable_ranges(
                transcript,
                video_title=source_info.get("title"),
                video_platform=source_info.get("platform"),
                video_duration=float(duration_seconds),
            )
            for candidate in ai_ranges:
                candidate["source"] = "ai"
        except Exception as exc:
            logger.warning("[%s] AI cleanup detection failed; using heuristic cleanup only: %s", job_id, exc)

        removal_intervals, trim_diagnostics = _combine_removal_ranges(
            heuristic_ranges=heuristic_ranges,
            ai_ranges=ai_ranges,
            duration_seconds=float(duration_seconds),
        )

        cleaned_video_path = source_video_path
        cleaned_storage_path = source_storage_path
        cleaned_transcript = dict(transcript)
        cleaned_duration_seconds = float(duration_seconds)

        if removal_intervals:
            _update_split_job_progress(job_id, 45, "rendering_clean_source", detail_key="removing_intro_outro_ad")
            keep_intervals = build_keep_intervals(
                window_start=0.0,
                window_end=float(duration_seconds),
                removal_intervals=removal_intervals,
            )
            if keep_intervals:
                timeline_map = build_timeline_map(keep_intervals)
                if timeline_map:
                    cleaned_output_path = os.path.join(work_dir, f"{video_id}_split_cleaned.mov")
                    render_condensed_video_from_keep_intervals(
                        input_video_path=source_video_path,
                        keep_intervals=keep_intervals,
                        output_path=cleaned_output_path,
                        work_dir=work_dir,
                    )
                    cleaned_video_path = cleaned_output_path
                    cleaned_duration_seconds = float(timeline_map[-1]["output_end"])
                    cleaned_transcript = _remap_transcript_to_clean_timeline(
                        transcript=transcript,
                        timeline_map=timeline_map,
                    )
                    cleaned_storage_path, _storage_etag = upload_raw_video(
                        video_id=video_id,
                        local_video_path=cleaned_video_path,
                        source_profile="source_split_cleaned",
                        logger=logger,
                        job_id=job_id,
                    )
                    trim_diagnostics["keep_intervals"] = [
                        [round(start, 3), round(end, 3)] for start, end in keep_intervals
                    ]
                    trim_diagnostics["timeline_map"] = timeline_map

        part_windows = _build_part_windows(
            duration_seconds=cleaned_duration_seconds,
            segment_length_seconds=segment_length_seconds,
            max_parts=_MAX_PARTS,
        )
        if not part_windows:
            raise RuntimeError("Batch split produced no valid output windows")

        _update_split_job_progress(job_id, 62, "creating_parts", detail_key="creating_split_parts")

        per_part_credits = int(
            calculate_custom_clip_generation_cost(False, subscription_tier)
        )
        actual_required_credits = len(part_windows) * per_part_credits
        if actual_required_credits > 0 and not has_sufficient_credits(
            user_id=billing_owner_user_id,
            amount=actual_required_credits,
            charge_source=charge_source,
            team_id=workspace_team_id,
        ):
            available = (
                get_team_wallet_balance(workspace_team_id)
                if charge_source == "team_wallet" and workspace_team_id
                else get_credit_balance(billing_owner_user_id)
            )
            raise RuntimeError(
                "Insufficient credits for split-video generation before fan-out: "
                f"required={actual_required_credits}, available={available}"
            )

        part_records: list[dict[str, Any]] = []
        title_prompt_segments: list[dict[str, Any]] = []
        for window in part_windows:
            part_transcript = _slice_transcript_for_window(
                transcript=cleaned_transcript,
                start_time=float(window["start"]),
                end_time=float(window["end"]),
            )
            part_records.append(
                {
                    "index": int(window["index"]),
                    "start": float(window["start"]),
                    "end": float(window["end"]),
                    "transcript": part_transcript,
                }
            )
            title_prompt_segments.append(
                {
                    "index": int(window["index"]),
                    "start": float(window["start"]),
                    "end": float(window["end"]),
                    "text": _title_prompt_text(part_transcript),
                }
            )

        generated_titles: list[str] = []
        try:
            generated_titles = analyzer.generate_segment_titles(
                title_prompt_segments,
                language_hint=(
                    str(cleaned_transcript.get("languageCode") or cleaned_transcript.get("language") or "").strip()
                    or source_detected_language
                ),
                video_title=str(source_info.get("title") or "").strip() or None,
            )
        except Exception as exc:
            logger.warning("[%s] AI title generation failed; using deterministic titles: %s", job_id, exc)

        for index, record in enumerate(part_records):
            ai_title = generated_titles[index] if index < len(generated_titles) else None
            record["title"] = _format_part_title(
                int(record["index"]),
                ai_title,
                str(source_info.get("title") or "").strip() or None,
            )

        clip_rows: list[dict[str, Any]] = []
        clip_generation_payloads: list[tuple[str, GenerateClipJob]] = []
        for record in part_records:
            clip_id = str(uuid4())
            clip_start = float(record["start"])
            clip_end = float(record["end"])
            clip_duration = max(1.0, clip_end - clip_start)
            quality_policy = resolve_clip_quality_policy(
                tier=subscription_tier,
                clip_duration_seconds=clip_duration,
                requested_output_quality=None,
            )
            clip_row = {
                "id": clip_id,
                "video_id": video_id,
                "user_id": user_id,
                "team_id": workspace_team_id,
                "billing_owner_user_id": billing_owner_user_id,
                "start_time": clip_start,
                "end_time": clip_end,
                "title": record["title"],
                "origin": "batch_split",
                "transcript": record["transcript"],
                "status": "pending",
                "source_video_storage_path_override": cleaned_storage_path,
            }
            if layout_id:
                clip_row["layout_id"] = layout_id
            clip_rows.append(clip_row)

            child_job_id = str(uuid4())
            child_job_data: GenerateClipJob = {
                "jobId": child_job_id,
                "clipId": clip_id,
                "userId": user_id,
                "layoutId": layout_id,
                "smartCleanupEnabled": False,
                "generationCredits": per_part_credits,
                "clipRetentionDays": clip_retention_days,
                "workspaceTeamId": workspace_team_id,
                "billingOwnerUserId": billing_owner_user_id,
                "chargeSource": charge_source,
                "workspaceRole": workspace_role,
                "subscriptionTier": subscription_tier,
                "sourceMaxHeight": quality_policy.get("source_max_height"),
                "sourceProfile": DEFAULT_SOURCE_PROFILE,
                "masterProfile": DEFAULT_MASTER_PROFILE,
                "deliveryProfile": DEFAULT_DELIVERY_PROFILE,
                "outputQualityOverride": None,
                "qualityPolicyProfile": quality_policy.get("profile"),
            }
            clip_generation_payloads.append((clip_id, child_job_data))

        clip_insert_response = supabase.table("clips").insert(clip_rows).execute()
        assert_response_ok(clip_insert_response, f"Failed to insert split clips for video {video_id}")

        _update_split_job_progress(
            job_id,
            80,
            "queueing_clip_generation",
            detail_key="queueing_generated_parts",
            detail_params={"count": len(clip_generation_payloads)},
        )

        child_job_ids: list[str] = []
        failed_child_job_ids: list[str] = []
        queue_name = _clip_queue_name(priority_processing)
        for clip_id, child_job_data in clip_generation_payloads:
            child_job_id, queued = _enqueue_generate_child_job(
                job_data=child_job_data,
                queue_name=queue_name,
                clip_id=clip_id,
                video_id=video_id,
                user_id=user_id,
                team_id=workspace_team_id,
                billing_owner_user_id=billing_owner_user_id,
            )
            child_job_ids.append(child_job_id)
            if not queued:
                failed_child_job_ids.append(child_job_id)

        result_data = {
            "stage": "completed",
            "expected_part_count": expected_part_count,
            "actual_part_count": len(part_records),
            "expected_generation_credits": expected_generation_credits,
            "actual_generation_credits_upper_bound": actual_required_credits,
            "created_clip_ids": [str(row["id"]) for row in clip_rows],
            "child_job_ids": child_job_ids,
            "failed_child_job_ids": failed_child_job_ids,
            "cleaned_source_storage_path": cleaned_storage_path,
            "trim_diagnostics": trim_diagnostics,
        }

        if manage_video_status:
            update_video_status(
                video_id,
                "completed",
                title=source_info.get("title"),
                thumbnail_url=source_info.get("thumbnail"),
                duration_seconds=duration_seconds,
                transcript=transcript,
            )

        update_job_status(job_id, "completed", 100, result_data=result_data)
    except Exception as exc:
        error_message = str(exc).strip() or "Batch split failed"
        logger.error("[%s] Batch split failed for video %s: %s", job_id, video_id, error_message)
        logger.debug("[%s] Batch split traceback:\n%s", job_id, traceback.format_exc())
        _best_effort_mark_failed(
            job_id=job_id,
            video_id=video_id,
            error_message=error_message,
            manage_video_status=manage_video_status,
        )
        raise
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

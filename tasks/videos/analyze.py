import logging
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any

from config import (
    WHISPER_FULL_TRANSCRIPT_WORD_TIMESTAMPS,
    calculate_video_analysis_cost,
)
from services.ai_analyzer import AIAnalyzer
from services.clips.render_profiles import DEFAULT_SOURCE_PROFILE
from services.transcriber import Transcriber
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.source_video import (
    build_raw_video_metadata_update,
    upload_raw_video_to_storage,
)
from tasks.models.jobs import AnalyzeVideoJob
from tasks.videos.source_transcript import resolve_source_transcript
from tasks.videos.transcript import normalize_transcript_for_analysis_window
from utils.sentry_context import configure_job_scope
from utils.workdirs import create_work_dir
from utils.supabase_client import (
    assert_response_ok,
    capture_credit_reservation,
    charge_video_analysis_credits,
    emit_video_analysis_usage_event,
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    has_team_wallet_charge_for_job,
    release_credit_reservation,
    reserve_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)

_MIN_CLIP_SECONDS = 10
_BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS = 0.75
_DEFAULT_LEGACY_CLIP_LENGTH_SECONDS = 90
try:
    _LOW_AI_SCORE_THRESHOLD = max(
        0.0,
        min(1.0, float(os.getenv("ANALYZER_MIN_AI_SCORE", "0.35"))),
    )
except (TypeError, ValueError):
    _LOW_AI_SCORE_THRESHOLD = 0.35

_INTRO_OUTRO_PATTERNS = (
    re.compile(r"\b(welcome\s+back|welcome\s+to\s+my\s+channel)\b", re.IGNORECASE),
    re.compile(r"\b(hey|hi|hello)\s+(guys|everyone|folks|friends)\b", re.IGNORECASE),
    re.compile(r"\b(in\s+this\s+(video|episode)|today\s+(i|we)'?re?\s+going\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(before\s+we\s+get\s+started|quick\s+announcement)\b", re.IGNORECASE),
    re.compile(r"\b(let'?s\s+(get\s+into|dive\s+in|jump\s+in))\b", re.IGNORECASE),
    re.compile(r"\b(thanks?\s+for\s+watching|see\s+you\s+in\s+the\s+next)\b", re.IGNORECASE),
    re.compile(r"\b(that'?s\s+it\s+for\s+today|hope\s+you\s+enjoyed)\b", re.IGNORECASE),
)

_NON_VIRAL_PATTERNS = (
    re.compile(r"\b(like\s+and\s+subscribe|hit\s+the\s+bell)\b", re.IGNORECASE),
    re.compile(
        r"\b(don'?t\s+forget\s+to|be\s+sure\s+to)\s+(like|subscribe|follow|comment|share)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(please\s+subscribe|go\s+subscribe|follow\s+me\s+on|follow\s+for\s+more|share\s+this\s+video)\b", re.IGNORECASE),
    re.compile(r"\b(link\s+in\s+(bio|description)|check\s+the\s+description)\b", re.IGNORECASE),
    re.compile(r"\b(this\s+video\s+is\s+sponsored\s+by|sponsored\s+by|sponsorship\s+from)\b", re.IGNORECASE),
    re.compile(r"\b(use\s+code\s+\w+|promo\s+code|affiliate\s+link)\b", re.IGNORECASE),
    re.compile(
        r"\b(quick\s+disclaimer|not\s+financial\s+advice|for\s+educational\s+purposes)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(let\s+me\s+know\s+in\s+the\s+comments)\b", re.IGNORECASE),
    re.compile(r"\b(apologies?\s+for\s+(the|any)|technical\s+difficult(y|ies)|audio\s+issue)\b", re.IGNORECASE),
)


def _resolve_legacy_clip_range(clip_length_seconds: int | None) -> tuple[int, int]:
    selected_max = max(
        _MIN_CLIP_SECONDS,
        int(clip_length_seconds or _DEFAULT_LEGACY_CLIP_LENGTH_SECONDS),
    )
    if selected_max <= 60:
        selected_min = _MIN_CLIP_SECONDS
    elif selected_max <= 90:
        selected_min = 60
    elif selected_max <= 120:
        selected_min = 90
    else:
        selected_min = 120

    selected_min = min(selected_min, selected_max)
    selected_min = max(_MIN_CLIP_SECONDS, selected_min)
    return selected_min, selected_max


def _resolve_requested_clip_range(job_data: AnalyzeVideoJob) -> tuple[int, int, int | None]:
    clip_length_seconds_raw = job_data.get("clipLengthSeconds")
    clip_length_seconds = (
        int(clip_length_seconds_raw) if clip_length_seconds_raw is not None else None
    )
    clip_length_min_raw = job_data.get("clipLengthMinSeconds")
    clip_length_max_raw = job_data.get("clipLengthMaxSeconds")

    if (clip_length_min_raw is None) != (clip_length_max_raw is None):
        raise RuntimeError(
            "Invalid clip-length payload: clipLengthMinSeconds and clipLengthMaxSeconds must be provided together."
        )

    if clip_length_min_raw is not None and clip_length_max_raw is not None:
        min_clip_seconds = int(clip_length_min_raw)
        max_clip_seconds = int(clip_length_max_raw)
    else:
        min_clip_seconds, max_clip_seconds = _resolve_legacy_clip_range(
            clip_length_seconds
        )

    if min_clip_seconds < _MIN_CLIP_SECONDS:
        raise RuntimeError(
            f"Invalid minimum clip length ({min_clip_seconds}s). Minimum allowed is {_MIN_CLIP_SECONDS}s."
        )
    if max_clip_seconds < min_clip_seconds:
        raise RuntimeError(
            "Invalid clip-length range: minimum cannot be greater than maximum."
        )

    selected_clip_length_seconds = (
        clip_length_seconds if clip_length_seconds is not None else max_clip_seconds
    )
    return min_clip_seconds, max_clip_seconds, selected_clip_length_seconds


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


def _nearest_boundary(
    *,
    value: float,
    candidates: list[float],
    tolerance: float,
) -> float | None:
    if not candidates:
        return None
    nearest = min(candidates, key=lambda candidate: abs(candidate - value))
    if abs(nearest - value) <= tolerance:
        return nearest
    return None


def _collect_clip_transcript_text(
    *,
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> str:
    parts: list[str] = []
    tolerance = _BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS
    for segment in segments:
        seg_start = _as_float(segment.get("start"))
        seg_end = _as_float(segment.get("end"))
        if seg_start is None or seg_end is None:
            continue
        if seg_start < start - tolerance or seg_end > end + tolerance:
            continue

        text = str(segment.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _looks_low_viral_potential(
    *,
    text: str,
    start: float,
    end: float,
    video_duration_seconds: float,
) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True

    if any(pattern.search(normalized) for pattern in _NON_VIRAL_PATTERNS):
        return True

    if video_duration_seconds <= 0:
        return any(pattern.search(normalized) for pattern in _INTRO_OUTRO_PATTERNS)

    edge_window = max(60.0, video_duration_seconds * 0.08)
    in_intro_or_outro_zone = (
        start <= edge_window
        or end >= max(0.0, video_duration_seconds - edge_window)
    )
    if in_intro_or_outro_zone and any(
        pattern.search(normalized) for pattern in _INTRO_OUTRO_PATTERNS
    ):
        return True

    return False


def _update_analysis_job_progress(
    job_id: str,
    progress: int,
    stage: str,
    billing_state: dict[str, Any] | None = None,
):
    """Persist progress percentage plus a machine-readable stage label."""
    result_data: dict[str, Any] = {"stage": stage}
    if billing_state:
        result_data["billing"] = dict(billing_state)
    update_job_status(
        job_id,
        "processing",
        progress,
        result_data=result_data,
    )


def _is_latest_analyze_job_for_video(*, job_id: str, video_id: str) -> bool:
    latest_job_resp = (
        supabase.table("jobs")
        .select("id")
        .eq("video_id", video_id)
        .eq("type", "analyze_video")
        .order("created_at", desc=True)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    assert_response_ok(latest_job_resp, f"Failed to load latest analyze job for video {video_id}")
    latest_jobs = latest_job_resp.data or []
    if not latest_jobs:
        return True

    latest_job_id = latest_jobs[0].get("id")
    return latest_job_id == job_id


def _best_effort_mark_superseded(*, job_id: str, video_id: str):
    reason = (
        "Superseded by a newer video analysis request for the same source. "
        "This job exited without charging credits."
    )
    try:
        update_job_status(
            job_id,
            "failed",
            0,
            reason,
            result_data={
                "stage": "superseded",
                "detail_key": "superseded_by_newer_request",
                "video_id": video_id,
            },
        )
    except Exception as exc:
        logger.warning("[%s] Failed to mark analyze job superseded: %s", job_id, exc)


def _best_effort_mark_failed(*, job_id: str, video_id: str, error_msg: str):
    try:
        update_job_status(job_id, "failed", 0, error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to update job failed status: %s", job_id, exc)

    try:
        update_video_status(video_id, "failed", error_message=error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to mark video %s failed: %s", job_id, video_id, exc)


def _has_retries_remaining() -> bool:
    try:
        from rq import get_current_job

        current = get_current_job()
        if current is None:
            return False
        return int(getattr(current, "retries_left", 0) or 0) > 0
    except Exception:
        return False


def _release_pending_analysis_reservation(
    *,
    job_id: str,
    charge_source: str,
    reservation_id: str | None,
    reservation_captured: bool,
    billing_state: dict[str, Any],
):
    if charge_source != "owner_wallet" or not reservation_id or reservation_captured:
        return

    try:
        release_credit_reservation(reservation_id=reservation_id)
        billing_state["status"] = "released"
    except Exception as exc:
        logger.warning(
            "[%s] Failed to release superseded analysis credit reservation %s: %s",
            job_id,
            reservation_id,
            exc,
        )


def _should_exit_for_newer_analyze_job(
    *,
    job_id: str,
    video_id: str,
    charge_source: str,
    reservation_id: str | None,
    reservation_captured: bool,
    billing_state: dict[str, Any],
) -> bool:
    if _is_latest_analyze_job_for_video(job_id=job_id, video_id=video_id):
        return False

    logger.info(
        "[%s] Skipping analysis for video %s because a newer job is active",
        job_id,
        video_id,
    )
    _release_pending_analysis_reservation(
        job_id=job_id,
        charge_source=charge_source,
        reservation_id=reservation_id,
        reservation_captured=reservation_captured,
        billing_state=billing_state,
    )
    _best_effort_mark_superseded(job_id=job_id, video_id=video_id)
    return True


def _count_video_clips(video_id: str) -> int:
    response = (
        supabase.table("clips")
        .select("id", count="exact")
        .eq("video_id", video_id)
        .execute()
    )
    assert_response_ok(response, f"Failed to count clips for {video_id}")
    count_value = getattr(response, "count", None)
    if isinstance(count_value, int):
        return count_value
    return len(response.data or [])


def _load_existing_video_transcript(video_id: str) -> dict[str, Any] | None:
    """Best-effort transcript reuse for re-analysis of an existing video row."""
    try:
        response = (
            supabase.table("videos")
            .select("transcript")
            .eq("id", video_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(response, f"Failed to load transcript for {video_id}")
    except Exception as exc:
        logger.warning("[%s] Failed to load existing transcript for reuse: %s", video_id, exc)
        return None

    data = response.data
    row = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else None)
    if not isinstance(row, dict):
        return None

    transcript = row.get("transcript")
    if isinstance(transcript, dict) and transcript.get("segments"):
        return transcript
    return None


def analyze_video_task(job_data: AnalyzeVideoJob):
    """Main task for video analysis.

    Each invocation gets its own working directory under ``TEMP_DIR`` so
    multiple workers can process different videos concurrently without
    file-path collisions.
    """
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    user_id = job_data["userId"]
    workspace_team_id = job_data.get("workspaceTeamId")
    billing_owner_user_id = job_data.get("billingOwnerUserId") or user_id
    charge_source = str(job_data.get("chargeSource") or "owner_wallet")
    url = job_data["url"]
    num_clips = job_data.get("numClips", 5)
    min_clip_seconds, max_clip_seconds, selected_clip_length_seconds = (
        _resolve_requested_clip_range(job_data)
    )
    processing_start_seconds = max(
        0.0,
        float(job_data.get("processingStartSeconds") or 0.0),
    )
    raw_processing_end_seconds = job_data.get("processingEndSeconds")
    processing_end_seconds: float | None = (
        float(raw_processing_end_seconds)
        if raw_processing_end_seconds is not None
        else None
    )
    extra_prompt = str(job_data.get("extraPrompt") or "").strip() or None
    expected_credits = int(job_data.get("analysisCredits") or 0)
    source_title = str(job_data.get("sourceTitle") or "").strip() or None
    source_thumbnail_url = str(job_data.get("sourceThumbnailUrl") or "").strip() or None
    source_platform = str(job_data.get("sourcePlatform") or "").strip().lower() or None
    source_external_id = str(job_data.get("sourceExternalId") or "").strip() or None
    source_detected_language = (
        str(job_data.get("sourceDetectedLanguage") or "").strip() or None
    )
    source_has_captions = _as_optional_bool(job_data.get("sourceHasCaptions"))
    source_has_audio = _as_optional_bool(job_data.get("sourceHasAudio"))

    configure_job_scope(
        job_id=job_id,
        job_type="analyze_video",
        user_id=user_id,
        video_id=video_id,
    )

    work_dir: str | None = None
    downloader: VideoDownloader | None = None
    analyzer: AIAnalyzer | None = None
    video_path: str | None = None
    source_media_path: str | None = None
    audio_path: str | None = None
    credits_required = expected_credits if expected_credits > 0 else 0
    duration_seconds = 0
    inserted_count = 0
    transcript_window_stats: dict[str, int] = {
        "segments_total": 0,
        "segments_used": 0,
        "segments_dropped_partial_without_words": 0,
        "segments_dropped_invalid": 0,
        "segments_clipped_with_words": 0,
        "segments_split_plain_text": 0,
    }
    clip_validation_stats: dict[str, int] = {
        "candidate_returned": 0,
        "candidate_validated": 0,
        "accepted": 0,
        "skipped_invalid_payload": 0,
        "skipped_outside_window": 0,
        "skipped_not_boundary_aligned": 0,
        "skipped_duration_out_of_range": 0,
        "skipped_low_score": 0,
        "skipped_non_viral_content": 0,
        "skipped_overlap": 0,
        "skipped_duplicate": 0,
    }
    transcript_diagnostics: dict[str, Any] = {
        "transcript_source": None,
        "transcript_language": None,
        "transcript_reused": False,
        "transcript_fallback_reason": None,
    }
    analysis_ai_diagnostics: dict[str, Any] = {
        "analysis_mode": "single_pass",
        "chunk_count": 1,
        "candidate_target": 0,
        "candidate_returned": 0,
    }

    billing_state: dict[str, Any] = {
        "mode": "owner_wallet_reservation"
        if charge_source == "owner_wallet"
        else "team_wallet",
        "status": "pending",
    }
    reservation_key = f"video_analysis:{job_id}"
    reservation_id: str | None = None
    reservation_captured = False
    team_wallet_already_charged = False
    existing_transcript: dict[str, Any] | None = None

    try:
        _update_analysis_job_progress(job_id, 0, "starting", billing_state=billing_state)

        if _should_exit_for_newer_analyze_job(
            job_id=job_id,
            video_id=video_id,
            charge_source=charge_source,
            reservation_id=reservation_id,
            reservation_captured=reservation_captured,
            billing_state=billing_state,
        ):
            return

        if expected_credits > 0 and not has_sufficient_credits(
            user_id=billing_owner_user_id,
            amount=expected_credits,
            charge_source=charge_source,
            team_id=workspace_team_id,
        ):
            available = (
                get_team_wallet_balance(workspace_team_id)
                if charge_source == "team_wallet" and workspace_team_id
                else get_credit_balance(billing_owner_user_id)
            )
            raise RuntimeError(
                "Insufficient credits for analysis before processing starts: "
                f"required={expected_credits}, available={available}"
            )

        # -- Per-job working directory for isolation ----------------------
        work_dir = create_work_dir(f"analyze_{job_id}")
        downloader = VideoDownloader(work_dir=work_dir)
        analyzer = AIAnalyzer()

        if charge_source == "owner_wallet" and expected_credits > 0:
            reservation_id = reserve_credits(
                user_id=billing_owner_user_id,
                amount=expected_credits,
                reason=f"Video analysis reservation ({job_id})",
                reservation_key=reservation_key,
                video_id=video_id,
            )
            if not reservation_id:
                raise RuntimeError(
                    "Insufficient credits for analysis before processing starts: "
                    f"required={expected_credits}"
                )
            billing_state.update(
                {
                    "reservation_key": reservation_key,
                    "reservation_id": reservation_id,
                    "status": "reserved",
                }
            )
            _update_analysis_job_progress(
                job_id,
                1,
                "reserving_credits",
                billing_state=billing_state,
            )

        # 1. Resolve source metadata (no full media download by default) -----
        _update_analysis_job_progress(
            job_id,
            5,
            "resolving_source",
            billing_state=billing_state,
        )

        source_info: dict[str, Any] = {
            "title": source_title,
            "thumbnail": source_thumbnail_url,
            "platform": source_platform,
            "external_id": source_external_id,
            "detected_language": source_detected_language,
            "has_captions": source_has_captions,
            "has_audio": source_has_audio,
            "duration": int(job_data.get("analysisDurationSeconds") or 0),
        }
        needs_probe = (
            int(source_info.get("duration") or 0) <= 0
            or not str(source_info.get("platform") or "").strip()
            or source_info.get("has_captions") is None
            or source_info.get("has_audio") is None
            or not str(source_info.get("detected_language") or "").strip()
        )
        if needs_probe:
            logger.info("[%s] Probing source metadata without media download ...", job_id)
            probe_data = downloader.probe_url(url)
            source_info["duration"] = int(
                source_info.get("duration") or probe_data.get("duration_seconds") or 0
            )
            source_info["title"] = source_info.get("title") or probe_data.get("title")
            source_info["thumbnail"] = source_info.get("thumbnail") or probe_data.get("thumbnail")
            source_info["platform"] = (
                str(source_info.get("platform") or probe_data.get("platform") or "")
                .strip()
                .lower()
            ) or None
            source_info["external_id"] = (
                str(source_info.get("external_id") or probe_data.get("external_id") or "")
                .strip()
            ) or None
            if source_info.get("has_captions") is None:
                source_info["has_captions"] = bool(probe_data.get("has_captions"))
            if source_info.get("has_audio") is None:
                source_info["has_audio"] = bool(probe_data.get("has_audio"))
            if not str(source_info.get("detected_language") or "").strip():
                source_info["detected_language"] = (
                    str(probe_data.get("detected_language") or "").strip() or None
                )

        _update_analysis_job_progress(
            job_id,
            20,
            "resolving_source",
            billing_state=billing_state,
        )

        # Calculate credits based on billed processing window.
        duration_seconds = int(source_info.get("duration") or 0)
        if duration_seconds <= 0:
            raise RuntimeError("Could not determine source video duration for analysis")

        processing_end_seconds = (
            min(float(duration_seconds), processing_end_seconds)
            if processing_end_seconds is not None
            else float(duration_seconds)
        )
        processing_start_seconds = min(processing_start_seconds, processing_end_seconds)
        if processing_end_seconds <= processing_start_seconds:
            raise RuntimeError(
                "Invalid processing timeframe: end must be greater than start "
                f"(start={processing_start_seconds}, end={processing_end_seconds})"
            )
        processing_window_seconds = processing_end_seconds - processing_start_seconds
        if processing_window_seconds < float(min_clip_seconds):
            raise RuntimeError(
                "Invalid processing timeframe: window is shorter than requested clip length "
                f"(window={processing_window_seconds:.2f}s, clip_length={min_clip_seconds}s)"
            )
        calculated_credits = calculate_video_analysis_cost(
            int(ceil(processing_window_seconds))
        )
        credits_required = expected_credits if expected_credits > 0 else calculated_credits
        if expected_credits > 0 and expected_credits != calculated_credits:
            logger.warning(
                "[%s] Analysis credit mismatch detected: queued=%d calculated=%d "
                "(window=%.2fs). Charging queued amount for deterministic billing.",
                job_id,
                expected_credits,
                calculated_credits,
                processing_window_seconds,
            )

        logger.info(
            "[%s] Video duration: %ds (~%.1f min), billed window: %.2fs (~%.1f min) - credits required: %d",
            job_id,
            duration_seconds,
            duration_seconds / 60,
            processing_window_seconds,
            processing_window_seconds / 60,
            credits_required,
        )

        if charge_source == "owner_wallet" and not reservation_id and credits_required > 0:
            reservation_id = reserve_credits(
                user_id=billing_owner_user_id,
                amount=credits_required,
                reason=f"Video analysis reservation ({job_id})",
                reservation_key=reservation_key,
                video_id=video_id,
            )
            if not reservation_id:
                available = get_credit_balance(billing_owner_user_id)
                raise RuntimeError(
                    "Insufficient credits for analysis before processing starts: "
                    f"required={credits_required}, available={available}"
                )
            billing_state.update(
                {
                    "reservation_key": reservation_key,
                    "reservation_id": reservation_id,
                    "status": "reserved",
                }
            )
            _update_analysis_job_progress(
                job_id,
                21,
                "reserving_credits",
                billing_state=billing_state,
            )

        # Update video metadata as soon as source metadata is available.
        update_video_status(
            video_id,
            "analyzing",
            title=source_info.get("title"),
            duration_seconds=duration_seconds,
            thumbnail_url=source_info.get("thumbnail"),
            platform=source_info.get("platform"),
            external_id=source_info.get("external_id"),
        )

        # 2. Get transcript ---------------------------------------------------
        _update_analysis_job_progress(
            job_id,
            35,
            "fetching_source_captions",
            billing_state=billing_state,
        )

        def _on_transcript_attempt(step: str):
            if step in {"youtube_captions", "provider_captions"}:
                _update_analysis_job_progress(
                    job_id,
                    35,
                    "fetching_source_captions",
                    billing_state=billing_state,
                )

        def _transcribe_with_whisper(language_hint: str | None) -> tuple[dict[str, Any], bool]:
            nonlocal source_media_path, video_path, audio_path, source_info
            if source_info.get("has_audio") is False:
                raise RuntimeError(
                    "Source does not expose downloadable audio and no source captions were found"
                )

            _update_analysis_job_progress(
                job_id,
                40,
                "downloading_source_media",
                billing_state=billing_state,
            )
            logger.info("[%s] Downloading source audio for Whisper transcription ...", job_id)
            try:
                audio_download = downloader.download_audio_only(url, video_id)
                source_media_path = audio_download["path"]
            except Exception as exc:
                logger.warning(
                    "[%s] Audio-only download failed for %s; falling back to full video download: %s",
                    job_id,
                    video_id,
                    exc,
                )
                full_video_data = downloader.download(url, video_id)
                video_path = full_video_data["path"]
                source_media_path = video_path
                source_info["title"] = source_info.get("title") or full_video_data.get("title")
                source_info["thumbnail"] = source_info.get("thumbnail") or full_video_data.get("thumbnail")
                source_info["platform"] = source_info.get("platform") or full_video_data.get("platform")
                source_info["external_id"] = source_info.get("external_id") or full_video_data.get("external_id")
                raw_video_metadata = build_raw_video_metadata_update(video_path)
                try:
                    raw_storage_path, raw_storage_etag = upload_raw_video_to_storage(
                        video_id=video_id,
                        local_video_path=video_path,
                        source_profile=DEFAULT_SOURCE_PROFILE,
                        logger=logger,
                        job_id=job_id,
                    )
                    raw_video_metadata["raw_video_storage_path"] = raw_storage_path
                    raw_video_metadata["raw_video_storage_etag"] = raw_storage_etag
                except Exception as raw_exc:
                    logger.warning(
                        "[%s] Canonical raw-source upload failed for %s: %s",
                        job_id,
                        video_id,
                        raw_exc,
                    )
                update_video_status(
                    video_id,
                    "analyzing",
                    title=source_info.get("title"),
                    thumbnail_url=source_info.get("thumbnail"),
                    platform=source_info.get("platform"),
                    external_id=source_info.get("external_id"),
                    **raw_video_metadata,
                )

            _update_analysis_job_progress(
                job_id,
                45,
                "extracting_audio",
                billing_state=billing_state,
            )
            logger.info("[%s] Extracting audio for Whisper transcription ...", job_id)
            audio_path = downloader.extract_audio(source_media_path or video_path)
            _update_analysis_job_progress(
                job_id,
                50,
                "extracting_audio",
                billing_state=billing_state,
            )

            _update_analysis_job_progress(
                job_id,
                55,
                "transcribing_audio",
                billing_state=billing_state,
            )
            logger.info("[%s] Transcribing with Whisper ...", job_id)
            transcriber = Transcriber()
            transcript_payload = transcriber.transcribe(
                audio_path,
                language_hint=language_hint,
                word_timestamps=WHISPER_FULL_TRANSCRIPT_WORD_TIMESTAMPS,
            )
            transcript_payload["source"] = "whisper"
            _update_analysis_job_progress(
                job_id,
                60,
                "transcribing_audio",
                billing_state=billing_state,
            )
            return transcript_payload, True

        existing_transcript = _load_existing_video_transcript(video_id)

        transcript_resolution = resolve_source_transcript(
            existing_transcript=existing_transcript,
            downloader=downloader,
            source_url=url,
            source_platform=str(source_info.get("platform") or "").strip().lower() or None,
            source_external_id=str(source_info.get("external_id") or "").strip() or None,
            source_detected_language=str(source_info.get("detected_language") or "").strip() or None,
            source_has_audio=_as_optional_bool(source_info.get("has_audio")),
            whisper_fallback=_transcribe_with_whisper,
            job_id=job_id,
            on_attempt=_on_transcript_attempt,
        )
        transcript = transcript_resolution.transcript
        transcript_diagnostics = dict(transcript_resolution.diagnostics)
        if transcript_diagnostics.get("transcript_source") != "whisper":
            _update_analysis_job_progress(
                job_id,
                60,
                "fetching_source_captions",
                billing_state=billing_state,
            )

        analysis_segments, transcript_window_stats = normalize_transcript_for_analysis_window(
            transcript=transcript,
            start_time=processing_start_seconds,
            end_time=processing_end_seconds,
        )
        if not analysis_segments:
            raise RuntimeError(
                "No transcript content found in the selected processing timeframe: "
                f"{processing_start_seconds:.2f}-{processing_end_seconds:.2f}s"
            )
        transcript_for_analysis = dict(transcript)
        transcript_for_analysis["segments"] = analysis_segments

        # 3. Analyze with OpenAI -----------------------------------------------
        _update_analysis_job_progress(
            job_id,
            70,
            "analyzing_transcript",
            billing_state=billing_state,
        )
        logger.info("[%s] Analysing for %d clips with OpenAI ...", job_id, num_clips)
        if hasattr(analyzer, "find_best_clips_detailed"):
            analysis_result = analyzer.find_best_clips_detailed(
                transcript_for_analysis,
                num_clips=num_clips,
                min_duration=min_clip_seconds,
                max_duration=max_clip_seconds,
                extra_prompt=extra_prompt,
                video_title=source_info.get("title"),
                video_platform=source_info.get("platform"),
                video_duration=float(duration_seconds),
            )
            clips = list(analysis_result.get("clips") or [])
            analysis_ai_diagnostics = {
                **analysis_ai_diagnostics,
                **dict(analysis_result.get("diagnostics") or {}),
            }
        else:
            clips = analyzer.find_best_clips(
                transcript_for_analysis,
                num_clips=num_clips,
                min_duration=min_clip_seconds,
                max_duration=max_clip_seconds,
                extra_prompt=extra_prompt,
                video_title=source_info.get("title"),
                video_platform=source_info.get("platform"),
                video_duration=float(duration_seconds),
            )
            analysis_ai_diagnostics["candidate_target"] = int(num_clips)
            analysis_ai_diagnostics["candidate_returned"] = len(clips)

        if _should_exit_for_newer_analyze_job(
            job_id=job_id,
            video_id=video_id,
            charge_source=charge_source,
            reservation_id=reservation_id,
            reservation_captured=reservation_captured,
            billing_state=billing_state,
        ):
            return

        _update_analysis_job_progress(
            job_id,
            90,
            "saving_results",
            billing_state=billing_state,
        )

        # 4. Save results -----------------------------------------------------
        logger.info("[%s] Saving results ...", job_id)
        save_video_resp = supabase.table("videos").update(
            {"transcript": transcript}
        ).eq("id", video_id).execute()
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        # Insert clip suggestions.
        MIN_CLIP_SECONDS = min_clip_seconds
        MAX_CLIP_SECONDS = max_clip_seconds
        boundary_start_candidates = sorted(
            {
                float(segment["start"])
                for segment in analysis_segments
                if _as_float(segment.get("start")) is not None
            }
        )
        boundary_end_candidates = sorted(
            {
                float(segment["end"])
                for segment in analysis_segments
                if _as_float(segment.get("end")) is not None
            }
        )

        accepted_ranges: list[tuple[float, float]] = []

        clip_validation_stats["candidate_returned"] = len(clips)
        existing_resp = (
            supabase.table("clips")
            .select("start_time,end_time,title")
            .eq("video_id", video_id)
            .execute()
        )
        assert_response_ok(existing_resp, f"Failed to load existing clips for {video_id}")
        existing_keys = {
            (
                round(float(row["start_time"]), 2),
                round(float(row["end_time"]), 2),
                (row.get("title") or "").strip().lower(),
            )
            for row in (existing_resp.data or [])
        }

        for clip in clips:
            if inserted_count >= max(1, int(num_clips)):
                break
            clip_validation_stats["candidate_validated"] += 1
            start = _as_float(clip.get("start"))
            end = _as_float(clip.get("end"))
            if start is None or end is None or end <= start:
                clip_validation_stats["skipped_invalid_payload"] += 1
                continue

            if (
                start < processing_start_seconds - _BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS
                or end > processing_end_seconds + _BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS
                or end > float(duration_seconds) + _BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS
            ):
                clip_validation_stats["skipped_outside_window"] += 1
                continue

            snapped_start = _nearest_boundary(
                value=float(start),
                candidates=boundary_start_candidates,
                tolerance=_BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS,
            )
            snapped_end = _nearest_boundary(
                value=float(end),
                candidates=boundary_end_candidates,
                tolerance=_BOUNDARY_ALIGNMENT_TOLERANCE_SECONDS,
            )
            if snapped_start is None or snapped_end is None or snapped_end <= snapped_start:
                clip_validation_stats["skipped_not_boundary_aligned"] += 1
                continue

            start = float(snapped_start)
            end = float(snapped_end)
            duration = end - start
            if duration < MIN_CLIP_SECONDS or duration > MAX_CLIP_SECONDS:
                clip_validation_stats["skipped_duration_out_of_range"] += 1
                continue

            # ai_score is DECIMAL(3,2); clamp to 0-1 to avoid numeric overflow
            raw_score = clip.get("score")
            if raw_score is not None:
                try:
                    score = max(0.0, min(1.0, float(raw_score)))
                except (TypeError, ValueError):
                    score = None
            else:
                score = None

            if score is not None and score < _LOW_AI_SCORE_THRESHOLD:
                clip_validation_stats["skipped_low_score"] += 1
                continue

            clip_transcript_text = _collect_clip_transcript_text(
                segments=analysis_segments,
                start=start,
                end=end,
            )
            if _looks_low_viral_potential(
                text=clip_transcript_text,
                start=start,
                end=end,
                video_duration_seconds=float(duration_seconds),
            ):
                clip_validation_stats["skipped_non_viral_content"] += 1
                continue

            has_heavy_overlap = False
            for acc_start, acc_end in accepted_ranges:
                overlap_start_val = max(start, acc_start)
                overlap_end_val = min(end, acc_end)
                if overlap_end_val > overlap_start_val:
                    overlap_len = overlap_end_val - overlap_start_val
                    shorter_dur = min(duration, acc_end - acc_start)
                    if shorter_dur > 0 and overlap_len / shorter_dur > 0.50:
                        has_heavy_overlap = True
                        break
            if has_heavy_overlap:
                clip_validation_stats["skipped_overlap"] += 1
                continue

            clip_key = (
                round(start, 2),
                round(end, 2),
                str(clip.get("title", "")).strip().lower(),
            )
            if clip_key in existing_keys:
                logger.info(
                    "[%s] Skipping duplicate clip suggestion %.2f-%.2f '%s'",
                    job_id,
                    start,
                    end,
                    clip.get("title", ""),
                )
                clip_validation_stats["skipped_duplicate"] += 1
                continue

            clip_insert_resp = supabase.table("clips").insert(
                {
                    "video_id": video_id,
                    "user_id": user_id,
                    "team_id": workspace_team_id,
                    "billing_owner_user_id": billing_owner_user_id,
                    "start_time": start,
                    "end_time": end,
                    "title": clip["title"],
                    "origin": "ai_suggested",
                    "ai_score": round(score, 2) if score is not None else None,
                    "transcript_excerpt": (
                        clip_transcript_text
                        if clip_transcript_text
                        else str(clip.get("text") or "").strip()
                    )[:500],
                    "status": "pending",
                }
            ).execute()
            assert_response_ok(
                clip_insert_resp,
                f"Failed to insert clip suggestion for {video_id} ({start}-{end})",
            )
            existing_keys.add(clip_key)
            accepted_ranges.append((start, end))
            inserted_count += 1
            clip_validation_stats["accepted"] += 1

        analysis_ai_diagnostics["candidate_validated"] = clip_validation_stats["candidate_validated"]
        analysis_ai_diagnostics["accepted_clip_count"] = inserted_count

        # Charge credits atomically at finalization time.
        _update_analysis_job_progress(
            job_id,
            95,
            "charging_credits",
            billing_state=billing_state,
        )

        charge_description = (
            f"Video analysis ({processing_window_seconds / 60:.1f}min window): "
            f'{(source_info.get("title") or "")[:50]}'
        )
        usage_metadata = {
            "analyses_count": 1,
            "video_duration_seconds": duration_seconds,
            "processing_window_seconds": processing_window_seconds,
            "requested_clip_count": num_clips,
            "requested_clip_length_seconds": selected_clip_length_seconds,
            "min_clip_seconds": min_clip_seconds,
            "max_clip_seconds": max_clip_seconds,
            "processing_start_seconds": processing_start_seconds,
            "processing_end_seconds": processing_end_seconds,
            "extra_prompt_provided": bool(extra_prompt),
            "suggested_clip_count": inserted_count,
            "transcript_window_stats": transcript_window_stats,
            "clip_validation_stats": clip_validation_stats,
            "transcript_source": transcript_diagnostics.get("transcript_source"),
            "transcript_language": transcript_diagnostics.get("transcript_language"),
            "analysis_mode": analysis_ai_diagnostics.get("analysis_mode"),
            "chunk_count": analysis_ai_diagnostics.get("chunk_count"),
            "candidate_target": analysis_ai_diagnostics.get("candidate_target"),
            "candidate_returned": analysis_ai_diagnostics.get("candidate_returned"),
            "candidate_validated": clip_validation_stats.get("candidate_validated"),
            "accepted_clip_count": inserted_count,
            "platform": source_info.get("platform"),
        }

        if charge_source == "owner_wallet":
            if credits_required <= 0:
                billing_state["status"] = "not_required"
            else:
                if not reservation_id:
                    raise RuntimeError("Missing credit reservation before capture")
                capture_ok = capture_credit_reservation(
                    reservation_id=reservation_id,
                    tx_type="video_analysis",
                    description=charge_description,
                    processing_ref=f"video_analysis:{job_id}",
                )
                if not capture_ok:
                    raise RuntimeError("Failed to capture video-analysis credit reservation")
                reservation_captured = True
                billing_state["status"] = "captured"
                emit_video_analysis_usage_event(
                    user_id=user_id,
                    amount=credits_required,
                    video_id=video_id,
                    charge_source=charge_source,
                    team_id=workspace_team_id,
                    billing_owner_user_id=billing_owner_user_id,
                    actor_user_id=user_id,
                    job_id=job_id,
                    usage_metadata=usage_metadata,
                )
        else:
            if charge_source == "team_wallet" and workspace_team_id and credits_required > 0:
                team_wallet_already_charged = has_team_wallet_charge_for_job(
                    team_id=workspace_team_id,
                    job_id=job_id,
                    owner_user_id=billing_owner_user_id,
                )

            if team_wallet_already_charged:
                billing_state["status"] = "already_charged"
                logger.info(
                    "[%s] Skipping duplicate team-wallet charge for already-charged job",
                    job_id,
                )
            else:
                charge_video_analysis_credits(
                    user_id=user_id,
                    amount=credits_required,
                    description=charge_description,
                    video_id=video_id,
                    charge_source=charge_source,
                    team_id=workspace_team_id,
                    billing_owner_user_id=billing_owner_user_id,
                    actor_user_id=user_id,
                    job_id=job_id,
                    processing_ref=f"video_analysis:{job_id}",
                    usage_metadata=usage_metadata,
                )
                billing_state["status"] = "charged"

        finalize_video_resp = supabase.table("videos").update(
            {
                "status": "completed",
                "credits_charged": credits_required,
            }
        ).eq("id", video_id).execute()
        assert_response_ok(
            finalize_video_resp,
            f"Failed to finalize video metadata for {video_id}",
        )

        final_clip_count = _count_video_clips(video_id)

        _update_analysis_job_progress(
            job_id,
            99,
            "finalizing",
            billing_state=billing_state,
        )
        update_job_status(
            job_id,
            "completed",
            100,
            result_data={
                "stage": "completed",
                "clip_count": final_clip_count,
                "new_clip_count": inserted_count,
                "duration_seconds": duration_seconds,
                "credits_charged": credits_required,
                "requested_clip_length_seconds": selected_clip_length_seconds,
                "min_clip_seconds": min_clip_seconds,
                "max_clip_seconds": max_clip_seconds,
                "processing_start_seconds": processing_start_seconds,
                "processing_end_seconds": processing_end_seconds,
                "extra_prompt_provided": bool(extra_prompt),
                "transcript_source": transcript_diagnostics.get("transcript_source"),
                "transcript_language": transcript_diagnostics.get("transcript_language"),
                "transcript_reused": transcript_diagnostics.get("transcript_reused"),
                "transcript_fallback_reason": transcript_diagnostics.get("transcript_fallback_reason"),
                "provider_track_source": transcript_diagnostics.get("provider_track_source"),
                "provider_track_ext": transcript_diagnostics.get("provider_track_ext"),
                "provider_track_language": transcript_diagnostics.get("provider_track_language"),
                "analysis_mode": analysis_ai_diagnostics.get("analysis_mode"),
                "chunk_count": analysis_ai_diagnostics.get("chunk_count"),
                "candidate_target": analysis_ai_diagnostics.get("candidate_target"),
                "candidate_returned": analysis_ai_diagnostics.get("candidate_returned"),
                "candidate_validated": clip_validation_stats.get("candidate_validated"),
                "accepted_clip_count": inserted_count,
                "transcript_window_stats": transcript_window_stats,
                "clip_validation_stats": clip_validation_stats,
                "billing": dict(billing_state),
            },
        )
        logger.info(
            "[%s] Video analysis completed - %d clips saved", job_id, inserted_count
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error analysing video: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        if _has_retries_remaining():
            try:
                update_job_status(
                    job_id,
                    "retrying",
                    0,
                    error_msg,
                    result_data={"stage": "retrying"},
                )
            except Exception as exc:
                logger.warning("[%s] Failed to update retrying status: %s", job_id, exc)
            raise

        if charge_source == "owner_wallet" and reservation_id and not reservation_captured:
            try:
                release_credit_reservation(reservation_id=reservation_id)
                billing_state["status"] = "released"
                _update_analysis_job_progress(
                    job_id,
                    96,
                    "releasing_credit_reservation",
                    billing_state=billing_state,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to release analysis credit reservation %s: %s",
                    job_id,
                    reservation_id,
                    exc,
                )

        _best_effort_mark_failed(job_id=job_id, video_id=video_id, error_msg=error_msg)
        raise

    finally:
        # Clean up audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug("[%s] Removed temp audio %s", job_id, audio_path)

        # Clean up work directory if video was successfully uploaded to storage.
        # If upload didn't happen, keep local files for clip generation until
        # the maintenance cleanup runs (raw_video_expires_at).
        if work_dir and os.path.isdir(work_dir):
            import shutil as _shutil_cleanup
            try:
                _shutil_cleanup.rmtree(work_dir, ignore_errors=True)
                logger.debug("[%s] Cleaned up work directory %s", job_id, work_dir)
            except Exception as cleanup_exc:
                logger.warning(
                    "[%s] Failed to clean work_dir %s: %s",
                    job_id, work_dir, cleanup_exc,
                )

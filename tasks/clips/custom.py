"""Custom clip task -- download, transcript, and generate in one shot.

Combines the download + transcript logic from ``analyze_video_task`` with the
layout-loading and rendering logic from ``generate_clip_task`` so the user can
go straight from a URL + timestamps to a finished clip.
"""

import logging
import os
import shutil
import traceback

from config import normalize_custom_clip_generation_credits
from services.clip_generator import ClipGenerator, compute_video_position
from services.clips.constants import canvas_size_for_aspect_ratio
from services.clips.quality_policy import resolve_effective_output_quality
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.captions import build_caption_ass, resolve_caption_style_mode
from tasks.clips.helpers.lifecycle import (
    asset_expires_at_iso,
    best_effort_cleanup_uploaded_artifacts,
    build_progress_result_data,
    parse_retention_days,
    update_clip_job_progress,
    upload_clip_with_replace,
)
from tasks.clips.helpers.layout import (
    load_layout_overrides,
    maybe_download_layout_background_image,
    maybe_download_media_files,
    resolve_effective_layout_id,
)
from tasks.clips.helpers.media import probe_video_size
from tasks.clips.helpers.quality_controls import resolve_quality_controls
from tasks.clips.helpers.smart_cleanup import apply_balanced_smart_cleanup
from tasks.clips.helpers.source_video import (
    build_raw_video_metadata_update,
    resolve_source_video,
)
from tasks.models.jobs import CustomClipJob
from tasks.models.layout import merge_layout_configs
from tasks.videos.source_transcript import resolve_source_transcript
from tasks.videos.transcript import (
    needs_whisper_retranscription,
    transcript_has_word_timing,
    transcript_has_word_timing_in_window,
    transcribe_clip_window_with_whisper,
)
from utils.sentry_context import configure_job_scope
from utils.workdirs import create_work_dir
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)
# Keep explicit media type in this module so durability tests can verify upload intent.
_UPLOAD_CONTENT_TYPE = {"content-type": "video/mp4"}


def _is_missing_action_column_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "action" in text
        and (
            "does not exist" in text
            or "schema cache" in text
            or "could not find" in text
            or "not found" in text
        )
    )


def _update_failed_clip_record(*, clip_id: str, payload: dict):
    try:
        response = (
            supabase.table("clips")
            .update(payload)
            .eq("id", clip_id)
            .execute()
        )
    except Exception as exc:
        if not _is_missing_action_column_error(exc):
            raise
        fallback_payload = dict(payload)
        fallback_payload.pop("action", None)
        fallback_response = (
            supabase.table("clips")
            .update(fallback_payload)
            .eq("id", clip_id)
            .execute()
        )
        assert_response_ok(
            fallback_response,
            f"Failed to mark clip {clip_id} failed (fallback without action)",
        )
        return

    try:
        assert_response_ok(response, f"Failed to mark clip {clip_id} failed")
    except Exception as exc:
        if not _is_missing_action_column_error(exc):
            raise
        fallback_payload = dict(payload)
        fallback_payload.pop("action", None)
        fallback_response = (
            supabase.table("clips")
            .update(fallback_payload)
            .eq("id", clip_id)
            .execute()
        )
        assert_response_ok(
            fallback_response,
            f"Failed to mark clip {clip_id} failed (fallback without action)",
        )


def _warn_low_source_resolution_for_high_output(
    *,
    job_id: str,
    video_id: str,
    output_quality: str,
    source_width: int,
    source_height: int,
    source_strategy: str,
) -> bool:
    if str(output_quality).strip().lower() != "high":
        return False
    if int(source_height) > 480:
        return False
    logger.warning(
        "[%s] High output quality requested but source video is low resolution "
        "(%dx%d via %s for %s); continuing with source constraints.",
        job_id,
        int(source_width),
        int(source_height),
        source_strategy,
        video_id,
    )
    return True


def _best_effort_mark_failed(
    *,
    job_id: str,
    clip_id: str,
    video_id: str,
    error_msg: str,
):
    try:
        update_job_status(job_id, "failed", 0, error_msg, action="retry")
    except Exception as exc:
        logger.warning("[%s] Failed to update job failed status: %s", job_id, exc)

    try:
        payload = {"status": "failed", "error_message": error_msg, "action": "retry"}
        _update_failed_clip_record(clip_id=clip_id, payload=payload)
    except Exception as exc:
        logger.warning("[%s] Failed to mark clip %s failed: %s", job_id, clip_id, exc)

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


def _load_video_context_row(video_id: str) -> dict:
    response = (
        supabase.table("videos")
        .select(
            "transcript,raw_video_path,raw_video_storage_path,url,title,duration_seconds,thumbnail_url,platform,external_id"
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


def _persist_video_raw_path(video_id: str, raw_video_path: str):
    payload = build_raw_video_metadata_update(raw_video_path)
    response = (
        supabase.table("videos")
        .update(payload)
        .eq("id", video_id)
        .execute()
    )
    assert_response_ok(response, f"Failed to persist raw video path for {video_id}")


def _persist_video_raw_metadata(video_id: str, payload: dict):
    response = (
        supabase.table("videos")
        .update(dict(payload))
        .eq("id", video_id)
        .execute()
    )
    assert_response_ok(response, f"Failed to persist raw video metadata for {video_id}")


def custom_clip_task(job_data: CustomClipJob):
    """Download a video, get its transcript, and generate a clip.

    ``job_data`` keys:
        jobId, videoId, clipId, userId, url, startTime, endTime, title,
        layoutId (optional, authoritative when valid for the user)
    """
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]
    url = job_data["url"]
    start_time = float(job_data["startTime"])
    end_time = float(job_data["endTime"])
    title = job_data["title"]
    layout_id: str | None = None
    layout_should_persist = False
    generation_credits = normalize_custom_clip_generation_credits(
        job_data.get("generationCredits")
    )
    clip_retention_days = parse_retention_days(job_data.get("clipRetentionDays"))
    smart_cleanup_enabled = bool(job_data.get("smartCleanupEnabled"))
    workspace_team_id = job_data.get("workspaceTeamId")
    billing_owner_user_id = job_data.get("billingOwnerUserId") or user_id
    charge_source = str(job_data.get("chargeSource") or "owner_wallet")
    workspace_role = str(job_data.get("workspaceRole") or "owner")
    source_max_height = job_data.get("sourceMaxHeight")
    try:
        source_max_height = int(source_max_height) if source_max_height is not None else None
    except (TypeError, ValueError):
        source_max_height = None
    output_quality_override = str(job_data.get("outputQualityOverride") or "").strip().lower() or None
    quality_policy_profile = str(job_data.get("qualityPolicyProfile") or "").strip() or None

    configure_job_scope(
        job_id=job_id,
        job_type="custom_clip",
        user_id=user_id,
        video_id=video_id,
        clip_id=clip_id,
    )

    smart_cleanup_summary = {
        "enabled": smart_cleanup_enabled,
        "profile": "balanced",
        "stopwords_removed": 0,
        "silence_seconds_removed": 0.0,
        "original_duration_seconds": 0.0,
        "output_duration_seconds": 0.0,
        "requested_window_start": float(start_time),
        "requested_window_end": float(end_time),
        "effective_window_start": float(start_time),
        "effective_window_end": float(end_time),
        "dropped_partial_words": 0,
    }

    # -- Per-job working directory ------------------------------------------
    work_dir = create_work_dir(f"custom_{clip_id}")

    downloader = VideoDownloader(work_dir=work_dir)
    audio_path: str | None = None
    uploaded_storage_path: str | None = None
    uploaded_file_size: int | None = None
    source_video_strategy = "reused_existing"
    source_video_wait_seconds = 0.0
    source_video_download_seconds = 0.0

    try:
        update_clip_job_progress(
            job_id=job_id,
            progress=0,
            stage="starting",
            detail_key="job_initialized",
        )

        if generation_credits > 0 and not has_sufficient_credits(
            user_id=billing_owner_user_id,
            amount=generation_credits,
            charge_source=charge_source,
            team_id=workspace_team_id,
        ):
            available = (
                get_team_wallet_balance(workspace_team_id)
                if charge_source == "team_wallet" and workspace_team_id
                else get_credit_balance(billing_owner_user_id)
            )
            raise RuntimeError(
                "Insufficient credits for custom clip generation before processing starts: "
                f"required={generation_credits}, available={available}"
            )

        update_video_status(video_id, "downloading")
        clip_status_resp = supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()
        assert_response_ok(clip_status_resp, f"Failed to mark clip {clip_id} generating")

        existing_video = _load_video_context_row(video_id)
        clip_layout_resp = (
            supabase.table("clips")
            .select("layout_id")
            .eq("id", clip_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(clip_layout_resp, f"Failed to load clip layout for {clip_id}")
        clip_rows = clip_layout_resp.data or []
        clip_layout_id = clip_rows[0].get("layout_id") if clip_rows else None

        layout_selection = resolve_effective_layout_id(
            user_id=user_id,
            workspace_team_id=workspace_team_id,
            workspace_role=workspace_role,
            job_id=job_id,
            logger=logger,
            requested_layout_id=job_data.get("layoutId"),
            clip_layout_id=clip_layout_id,
        )
        layout_id = layout_selection.layout_id
        layout_should_persist = layout_selection.should_persist_to_clip
        layout_overrides = load_layout_overrides(
            user_id=user_id,
            workspace_team_id=workspace_team_id,
            workspace_role=workspace_role,
            layout_id=layout_id,
            job_id=job_id,
            logger=logger,
        )
        bg_style = layout_overrides.bg_style
        bg_color = layout_overrides.bg_color
        blur_strength = layout_overrides.blur_strength
        output_quality = resolve_effective_output_quality(
            template_quality=layout_overrides.output_quality,
            policy_override_quality=output_quality_override,
        )
        quality_controls = resolve_quality_controls(
            output_quality=output_quality,
            policy_source_max_height=source_max_height,
        )
        logger.info(
            "[%s] Quality controls resolved (output=%s profile=%s policy_max_height=%s "
            "effective_max_height=%s fresh_download=%s upload_reencode=%s smart_cleanup=%s/%s)",
            job_id,
            output_quality,
            quality_policy_profile or "none",
            source_max_height if source_max_height is not None else "best",
            (
                quality_controls.effective_source_max_height
                if quality_controls.effective_source_max_height is not None
                else "best"
            ),
            quality_controls.prefer_fresh_source_download,
            quality_controls.allow_upload_reencode,
            quality_controls.smart_cleanup_crf,
            quality_controls.smart_cleanup_preset,
        )

        vid_cfg, title_cfg, cap_cfg, intro_cfg, outro_cfg, overlay_cfg = merge_layout_configs(
            layout_overrides.layout_video,
            layout_overrides.layout_title,
            layout_overrides.layout_captions,
            layout_overrides.layout_intro,
            layout_overrides.layout_outro,
            layout_overrides.layout_overlay,
        )
        canvas_aspect_ratio = str(vid_cfg.get("canvasAspectRatio") or "9:16")
        video_scale_mode = str(vid_cfg.get("videoScaleMode") or "fit")
        canvas_w, canvas_h = canvas_size_for_aspect_ratio(canvas_aspect_ratio)

        # 1. Resolve source video ---------------------------------------------
        waiting_stage_emitted = False

        def _emit_waiting_for_source(elapsed_seconds: float):
            nonlocal waiting_stage_emitted
            if waiting_stage_emitted:
                return
            waiting_stage_emitted = True
            update_clip_job_progress(
                job_id=job_id,
                progress=12,
                stage="waiting_for_source_video_download",
                detail_key="waiting_for_source_video",
                detail_params={"seconds": round(float(elapsed_seconds), 1)},
            )

        def _emit_downloading_source():
            update_clip_job_progress(
                job_id=job_id,
                progress=12,
                stage="downloading_video",
                detail_key="downloading_source_video",
            )

        source_resolution = resolve_source_video(
            video_id=video_id,
            source_url=url or existing_video.get("url"),
            initial_raw_video_path=existing_video.get("raw_video_path"),
            initial_raw_video_storage_path=existing_video.get("raw_video_storage_path"),
            downloader=downloader,
            load_video_row=_load_video_context_row,
            persist_raw_video_path=_persist_video_raw_path,
            persist_raw_video_metadata=_persist_video_raw_metadata,
            logger=logger,
            job_id=job_id,
            source_max_height=quality_controls.effective_source_max_height,
            prefer_fresh_download=quality_controls.prefer_fresh_source_download,
            on_wait_for_download=_emit_waiting_for_source,
            on_download_start=_emit_downloading_source,
        )
        video_path = source_resolution.video_path
        src_w = int(source_resolution.width)
        src_h = int(source_resolution.height)
        source_video_strategy = str(source_resolution.strategy)
        source_video_wait_seconds = float(source_resolution.wait_seconds)
        source_video_download_seconds = float(source_resolution.download_seconds)
        update_clip_job_progress(
            job_id=job_id,
            progress=20,
            stage="downloading_video",
            detail_key="source_video_ready",
        )
        logger.info(
            "[%s] Source video resolved for %s via %s (wait=%.2fs, download=%.2fs, max_height=%s, quality_profile=%s)",
            job_id,
            video_id,
            source_video_strategy,
            source_video_wait_seconds,
            source_video_download_seconds,
            (
                quality_controls.effective_source_max_height
                if quality_controls.effective_source_max_height is not None
                else "best"
            ),
            quality_policy_profile or "none",
        )
        _warn_low_source_resolution_for_high_output(
            job_id=job_id,
            video_id=video_id,
            output_quality=output_quality,
            source_width=src_w,
            source_height=src_h,
            source_strategy=source_video_strategy,
        )

        download_metadata = source_resolution.download_metadata or {}
        duration_seconds = int(
            download_metadata.get("duration")
            or existing_video.get("duration_seconds")
            or max(1.0, end_time + 10.0, end_time - start_time)
        )
        source_title = str(
            download_metadata.get("title")
            or existing_video.get("title")
            or title
        )
        source_thumbnail = download_metadata.get("thumbnail") or existing_video.get(
            "thumbnail_url"
        )
        source_platform = str(
            download_metadata.get("platform")
            or existing_video.get("platform")
            or "unknown"
        ).lower()
        source_external_id = (
            download_metadata.get("external_id")
            or existing_video.get("external_id")
        )
        raw_video_metadata = build_raw_video_metadata_update(video_path)
        if source_resolution.storage_path:
            raw_video_metadata["raw_video_storage_path"] = source_resolution.storage_path
        if source_resolution.storage_etag:
            raw_video_metadata["raw_video_storage_etag"] = source_resolution.storage_etag

        # Update video row with metadata
        update_video_status(
            video_id,
            "analyzing",
            title=source_title,
            duration_seconds=duration_seconds,
            thumbnail_url=source_thumbnail,
            platform=source_platform,
            external_id=source_external_id,
            raw_video_path=raw_video_metadata["raw_video_path"],
            raw_video_expires_at=raw_video_metadata["raw_video_expires_at"],
        )

        # 2. Get transcript ----------------------------------------------------
        transcript = None
        partial_transcript = False
        existing_transcript = existing_video.get("transcript")
        resolved_full_video_transcript = (
            existing_transcript
            if isinstance(existing_transcript, dict) and existing_transcript.get("segments")
            else None
        )

        update_clip_job_progress(
            job_id=job_id,
            progress=30,
            stage="fetching_source_captions",
            detail_key="checking_existing_captions",
        )

        def _on_transcript_attempt(step: str):
            if step in {"youtube_captions", "provider_captions"}:
                update_clip_job_progress(
                    job_id=job_id,
                    progress=30,
                    stage="fetching_source_captions",
                    detail_key="checking_existing_captions",
                )

        def _partial_whisper_fallback(language_hint: str | None) -> tuple[dict, bool]:
            update_clip_job_progress(
                job_id=job_id,
                progress=30,
                stage="fetching_source_captions",
                detail_key="checking_existing_captions",
            )
            update_clip_job_progress(
                job_id=job_id,
                progress=30,
                stage="extracting_audio",
                detail_key="preparing_audio_track",
            )
            logger.info("[%s] Extracting audio for Whisper transcription ...", job_id)
            audio_path = downloader.extract_audio(video_path)
            update_clip_job_progress(
                job_id=job_id,
                progress=38,
                stage="extracting_audio",
                detail_key="preparing_audio_track",
            )

            update_clip_job_progress(
                job_id=job_id,
                progress=45,
                stage="transcribing_audio",
                detail_key="creating_captions_from_speech",
            )
            transcript = transcribe_clip_window_with_whisper(
                media_path=audio_path,
                work_dir=work_dir,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                video_duration_seconds=duration_seconds,
                language_hint=language_hint,
            )
            return transcript, False

        transcript_resolution = resolve_source_transcript(
            existing_transcript=resolved_full_video_transcript,
            downloader=downloader,
            source_url=url,
            source_platform=source_platform,
            source_external_id=str(source_external_id or "").strip() or None,
            source_detected_language=None,
            source_has_audio=None,
            whisper_fallback=_partial_whisper_fallback,
            job_id=job_id,
            on_attempt=_on_transcript_attempt,
        )
        transcript = transcript_resolution.transcript
        partial_transcript = not transcript_resolution.is_full_transcript
        if transcript_resolution.is_full_transcript:
            resolved_full_video_transcript = transcript_resolution.transcript

        caption_style = resolve_caption_style_mode(cap_cfg)
        has_word_timing_for_window = transcript_has_word_timing_in_window(
            transcript,
            start_time=start_time,
            end_time=end_time,
            minimum_words=1,
        )
        should_retranscribe_for_captions = needs_whisper_retranscription(
            transcript,
            caption_style,
        )
        should_retranscribe_for_smart_cleanup = (
            smart_cleanup_enabled and not has_word_timing_for_window
        )
        if smart_cleanup_enabled and has_word_timing_for_window:
            logger.info(
                "[%s] Smart Cleanup reusing existing transcript word timings for %.2f-%.2fs",
                job_id,
                start_time,
                end_time,
            )

        if should_retranscribe_for_captions or should_retranscribe_for_smart_cleanup:
            retranscribe_reason = (
                "Smart Cleanup needs word timing in selected window"
                if should_retranscribe_for_smart_cleanup
                else f"caption style '{caption_style}' requires word timing"
            )
            update_clip_job_progress(
                job_id=job_id,
                progress=45,
                stage="retranscribing_with_whisper",
                detail_key="improving_caption_timing",
            )
            logger.info(
                "[%s] Re-transcribing clip segment with Whisper (%s)",
                job_id,
                retranscribe_reason,
            )
            transcript = transcribe_clip_window_with_whisper(
                media_path=video_path,
                work_dir=work_dir,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                video_duration_seconds=duration_seconds,
            )
            partial_transcript = True
            update_clip_job_progress(
                job_id=job_id,
                progress=52,
                stage="loading_layout",
                detail_key="applying_template_settings",
            )

        if smart_cleanup_enabled and not transcript_has_word_timing(transcript):
            raise RuntimeError(
                "Smart Cleanup requires Whisper word-level timings, but no usable words were returned."
            )

        update_clip_job_progress(
            job_id=job_id,
            progress=52,
            stage="loading_layout",
            detail_key="applying_template_settings",
        )

        # Save transcript on video row only when it represents the full video.
        video_update = {"status": "completed"}
        if resolved_full_video_transcript:
            video_update["transcript"] = resolved_full_video_transcript

        save_video_resp = (
            supabase.table("videos").update(video_update).eq("id", video_id).execute()
        )
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        bg_style, bg_image_path = maybe_download_layout_background_image(
            bg_style=bg_style,
            bg_image_storage_path=layout_overrides.bg_image_storage_path,
            work_dir=work_dir,
            job_id=job_id,
            logger=logger,
        )

        intro_file_path, outro_file_path, overlay_file_path = maybe_download_media_files(
            intro_cfg=intro_cfg,
            outro_cfg=outro_cfg,
            overlay_cfg=overlay_cfg,
            work_dir=work_dir,
            job_id=job_id,
            logger=logger,
        )

        if smart_cleanup_enabled:
            update_clip_job_progress(
                job_id=job_id,
                progress=60,
                stage="applying_smart_cleanup",
                detail_key="smart_cleanup_balanced",
            )
            cleanup_result = apply_balanced_smart_cleanup(
                transcript=transcript,
                video_path=video_path,
                clip_id=clip_id,
                work_dir=work_dir,
                start_time=start_time,
                end_time=end_time,
                crf=quality_controls.smart_cleanup_crf,
                preset=quality_controls.smart_cleanup_preset,
            )
            video_path = cleanup_result["video_path"]
            transcript = cleanup_result["transcript"]
            summary = cleanup_result["summary"]
            smart_cleanup_summary = {
                "enabled": True,
                "profile": str(summary.get("profile", "balanced")),
                "stopwords_removed": int(summary.get("stopwords_removed", 0)),
                "silence_seconds_removed": float(
                    summary.get("silence_seconds_removed", 0.0)
                ),
                "original_duration_seconds": float(
                    summary.get("original_duration_seconds", 0.0)
                ),
                "output_duration_seconds": float(
                    summary.get("output_duration_seconds", 0.0)
                ),
                "requested_window_start": float(
                    summary.get("requested_window_start", start_time)
                ),
                "requested_window_end": float(
                    summary.get("requested_window_end", end_time)
                ),
                "effective_window_start": float(
                    summary.get("effective_window_start", start_time)
                ),
                "effective_window_end": float(
                    summary.get("effective_window_end", end_time)
                ),
                "dropped_partial_words": int(summary.get("dropped_partial_words", 0)),
            }
            start_time = 0.0
            end_time = float(smart_cleanup_summary["output_duration_seconds"])

        # 4. Compute video position for caption placement ----------------------
        src_w, src_h = probe_video_size(video_path)
        _, vid_h, _, vid_y = compute_video_position(
            src_w,
            src_h,
            vid_cfg["widthPct"],
            vid_cfg["positionY"],
            vid_cfg.get("customX"),
            vid_cfg.get("customY"),
            vid_cfg.get("customWidth"),
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            video_scale_mode=video_scale_mode,
        )

        update_clip_job_progress(
            job_id=job_id,
            progress=65,
            stage="preparing_captions",
            detail_key="preparing_caption_text",
        )
        caption_ass_path = build_caption_ass(
            job_id=job_id,
            clip_id=clip_id,
            transcript=transcript,
            cap_cfg=cap_cfg,
            start_time=start_time,
            end_time=end_time,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            vid_y=vid_y,
            vid_h=vid_h,
            video_aspect_ratio=canvas_aspect_ratio,
            work_dir=work_dir,
            logger=logger,
        )

        # 6. Generate clip -----------------------------------------------------
        update_clip_job_progress(
            job_id=job_id,
            progress=75,
            stage="rendering_clip",
            detail_key="building_video",
            detail_params={
                "start_seconds": round(start_time, 2),
                "end_seconds": round(end_time, 2),
            },
        )
        generator = ClipGenerator(work_dir=work_dir)
        logger.info("[%s] Generating clip %s ...", job_id, clip_id)

        result = generator.generate(
            video_path=video_path,
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            title=title,
            background_style=bg_style,
            background_color=bg_color,
            background_image_path=bg_image_path,
            # video layout
            video_width_pct=vid_cfg["widthPct"],
            video_position_y=vid_cfg["positionY"],
            video_custom_x=vid_cfg.get("customX"),
            video_custom_y=vid_cfg.get("customY"),
            video_custom_width=vid_cfg.get("customWidth"),
            canvas_aspect_ratio=canvas_aspect_ratio,
            video_scale_mode=video_scale_mode,
            # title style
            title_show=title_cfg["show"],
            title_font_size=title_cfg["fontSize"],
            title_font_color=title_cfg["fontColor"],
            title_font_family=title_cfg["fontFamily"],
            title_align=title_cfg["align"],
            title_stroke_width=title_cfg["strokeWidth"],
            title_stroke_color=title_cfg["strokeColor"],
            title_bar_enabled=title_cfg["barEnabled"],
            title_bar_color=title_cfg["barColor"],
            title_padding_x=title_cfg["paddingX"],
            title_position_y=title_cfg["positionY"],
            title_custom_x=title_cfg.get("customX"),
            title_custom_y=title_cfg.get("customY"),
            title_custom_width=title_cfg.get("customWidth"),
            # captions
            caption_ass_path=caption_ass_path,
            # intro / outro / overlay
            intro_cfg=intro_cfg,
            outro_cfg=outro_cfg,
            overlay_cfg=overlay_cfg,
            intro_file_path=intro_file_path,
            outro_file_path=outro_file_path,
            overlay_file_path=overlay_file_path,
            # reframe (speaker tracking)
            reframe_enabled=bool(vid_cfg.get("reframeEnabled", False)),
            reframe_smoothing=float(vid_cfg.get("reframeSmoothing", 0.08)),
            reframe_center_bias=float(vid_cfg.get("reframeCenterBias", 0.6)),
            # misc
            blur_strength=blur_strength,
            output_quality=output_quality,
        )

        # 7. Upload to Supabase Storage ----------------------------------------
        storage_path = f"clips/{clip_id}.mp4"

        update_clip_job_progress(
            job_id=job_id,
            progress=84,
            stage="uploading_clip",
            detail_key="saving_generated_clip",
        )
        logger.info("[%s] Uploading clip to storage ...", job_id)

        uploaded_file_size = upload_clip_with_replace(
            local_clip_path=result["clip_path"],
            storage_path=storage_path,
            job_id=job_id,
            logger=logger,
            allow_reencode=quality_controls.allow_upload_reencode,
        )
        uploaded_storage_path = storage_path

        # 8. Update clip record ------------------------------------------------
        update_clip_job_progress(
            job_id=job_id,
            progress=90,
            stage="finalizing",
            detail_key="finalizing_clip_record",
        )
        clip_update = {
            "status": "completed",
            "storage_path": storage_path,
            "file_size_bytes": uploaded_file_size or result["file_size"],
            "asset_expires_at": asset_expires_at_iso(clip_retention_days),
            "asset_expired_at": None,
        }
        if layout_should_persist and layout_id:
            clip_update["layout_id"] = layout_id
        clip_update_resp = (
            supabase.table("clips").update(clip_update).eq("id", clip_id).execute()
        )
        assert_response_ok(clip_update_resp, f"Failed to finalize clip {clip_id}")

        # 9. Charge credits after finalizing completion state ------------------
        update_clip_job_progress(
            job_id=job_id,
            progress=97,
            stage="charging_credits",
            detail_key="updating_usage_records",
        )
        charge_clip_generation_credits(
            user_id=user_id,
            amount=generation_credits,
            description=f"Custom clip: {title[:50]}",
            video_id=video_id,
            clip_id=clip_id,
            charge_source=charge_source,
            team_id=workspace_team_id,
            billing_owner_user_id=billing_owner_user_id,
            actor_user_id=user_id,
            job_id=job_id,
            processing_ref=f"clip_generation:{job_id}",
            usage_metadata={
                "units_generated": 1,
                "smart_cleanup_enabled": smart_cleanup_enabled,
                "clip_duration_seconds": max(0.0, end_time - start_time),
                "workspace_role": workspace_role,
                "generation_flow": "custom",
            },
        )

        update_job_status(
            job_id,
            "completed",
            100,
            result_data=build_progress_result_data(
                stage="completed",
                detail_key="generation_finished",
                extra_result_data={
                    "storage_path": storage_path,
                    "file_size": uploaded_file_size or result["file_size"],
                    "source_video_strategy": source_video_strategy,
                    "source_video_wait_seconds": round(source_video_wait_seconds, 3),
                    "source_video_download_seconds": round(
                        source_video_download_seconds, 3
                    ),
                    "smart_cleanup": smart_cleanup_summary,
                },
            ),
        )
        logger.info("[%s] Custom clip completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating custom clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        best_effort_cleanup_uploaded_artifacts(
            job_id=job_id,
            clip_id=clip_id,
            storage_path=uploaded_storage_path,
            logger=logger,
        )
        if _has_retries_remaining():
            logger.warning(
                "[%s] Retries remain but auto-retry is disabled for custom clip failures; "
                "marking failed with retry action.",
                job_id,
            )

        _best_effort_mark_failed(
            job_id=job_id,
            clip_id=clip_id,
            video_id=video_id,
            error_msg=error_msg,
        )

        raise

    finally:
        # Clean up audio if extracted
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        # Remove the entire per-clip working directory
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

import logging
import os
import shutil
import traceback
from datetime import datetime, timedelta, timezone

from config import normalize_clip_generation_credits
from services.clip_generator import ClipGenerator, compute_video_position
from services.clips.constants import canvas_size_for_aspect_ratio
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.captions import build_caption_ass, resolve_caption_style_mode
from tasks.clips.helpers.smart_cleanup import apply_balanced_smart_cleanup
from tasks.clips.helpers.source_video import (
    build_raw_video_metadata_update,
    resolve_source_video,
)
from tasks.videos.transcript import (
    needs_whisper_retranscription,
    transcript_has_word_timing,
    transcript_has_word_timing_in_window,
    transcribe_clip_window_with_whisper,
)
from tasks.clips.helpers.layout import (
    load_layout_overrides,
    maybe_download_layout_background_image,
    maybe_download_media_files,
    resolve_effective_layout_id,
)
from tasks.models.jobs import GenerateClipJob
from tasks.models.layout import merge_layout_configs
from utils.workdirs import create_work_dir
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
    update_job_status,
)

logger = logging.getLogger(__name__)

_VIDEO_UPLOAD_OPTIONS = {"content-type": "video/mp4", "cache-control": "3600"}


def _parse_retention_days(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _asset_expires_at_iso(retention_days: int | None) -> str | None:
    if retention_days is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=retention_days)).isoformat()


def _is_duplicate_storage_error(exc: Exception) -> bool:
    payload = exc.args[0] if getattr(exc, "args", None) else None
    if isinstance(payload, dict):
        status_code = payload.get("statusCode")
        error_name = str(payload.get("error") or "").lower()
        message = str(payload.get("message") or "").lower()
        if status_code == 400 and (
            error_name == "duplicate" or "already exists" in message
        ):
            return True

    text = str(exc).lower()
    return "duplicate" in text or "already exists" in text


def _upload_clip_with_replace(
    *,
    local_clip_path: str,
    storage_path: str,
    job_id: str,
):
    with open(local_clip_path, "rb") as file_obj:
        try:
            supabase.storage.from_("generated-clips").upload(
                storage_path,
                file_obj,
                file_options=_VIDEO_UPLOAD_OPTIONS,
            )
            return
        except Exception as exc:
            if not _is_duplicate_storage_error(exc):
                raise

            logger.warning(
                "[%s] Storage path %s already exists. Replacing existing artifact.",
                job_id,
                storage_path,
            )
            supabase.storage.from_("generated-clips").remove([storage_path])
            file_obj.seek(0)
            supabase.storage.from_("generated-clips").upload(
                storage_path,
                file_obj,
                file_options=_VIDEO_UPLOAD_OPTIONS,
            )


def _is_latest_generate_job_for_clip(*, job_id: str, clip_id: str) -> bool:
    latest_job_resp = (
        supabase.table("jobs")
        .select("id")
        .eq("clip_id", clip_id)
        .eq("type", "generate_clip")
        .order("created_at", desc=True)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    assert_response_ok(latest_job_resp, f"Failed to load latest generate job for clip {clip_id}")
    latest_jobs = latest_job_resp.data or []
    if not latest_jobs:
        return True

    latest_job_id = latest_jobs[0].get("id")
    return latest_job_id == job_id


def _best_effort_mark_superseded(*, job_id: str, clip_id: str):
    reason = (
        "Superseded by a newer clip generation request for the same clip. "
        "This job exited without charging credits."
    )
    try:
        update_job_status(
            job_id,
            "failed",
            0,
            reason,
            result_data={"stage": "superseded", "clip_id": clip_id},
        )
    except Exception as exc:
        logger.warning("[%s] Failed to mark job superseded: %s", job_id, exc)


def _update_clip_job_progress(job_id: str, progress: int, stage: str):
    """Persist clip-generation progress plus machine-readable stage."""
    update_job_status(
        job_id,
        "processing",
        progress,
        result_data={"stage": stage},
    )


def _best_effort_cleanup_uploaded_artifacts(
    *,
    job_id: str,
    clip_id: str,
    storage_path: str | None,
):
    """Delete uploaded files and clear DB pointers after partial failure."""
    if storage_path:
        try:
            supabase.storage.from_("generated-clips").remove([storage_path])
        except Exception as exc:
            logger.warning(
                "[%s] Failed to delete uploaded clip artifact %s: %s",
                job_id,
                storage_path,
                exc,
            )

    if storage_path:
        try:
            clear_resp = (
                supabase.table("clips")
                .update(
                    {
                        "storage_path": None,
                        "thumbnail_path": None,
                        "file_size_bytes": None,
                        "asset_expires_at": None,
                        "asset_expired_at": None,
                    }
                )
                .eq("id", clip_id)
                .execute()
            )
            assert_response_ok(clear_resp, f"Failed to clear storage fields for {clip_id}")
        except Exception as exc:
            logger.warning(
                "[%s] Failed to clear clip storage pointers for %s: %s",
                job_id,
                clip_id,
                exc,
            )


def _best_effort_mark_failed(*, job_id: str, clip_id: str, error_msg: str):
    try:
        update_job_status(job_id, "failed", 0, error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to update job failed status: %s", job_id, exc)

    try:
        fail_resp = (
            supabase.table("clips")
            .update({"status": "failed", "error_message": error_msg})
            .eq("id", clip_id)
            .execute()
        )
        assert_response_ok(fail_resp, f"Failed to mark clip {clip_id} failed")
    except Exception as exc:
        logger.warning("[%s] Failed to mark clip %s failed: %s", job_id, clip_id, exc)


def _load_video_source_row(video_id: str) -> dict:
    response = (
        supabase.table("videos")
        .select("raw_video_path,url")
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


def generate_clip_task(job_data: GenerateClipJob):
    """Generate final clip with overlay.

    Each invocation creates a per-clip working directory so multiple
    clip-generation workers can run concurrently without collisions.
    The entire directory is removed in the ``finally`` block.
    """
    job_id = job_data["jobId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]
    layout_id: str | None = None
    layout_should_persist = False
    generation_credits = normalize_clip_generation_credits(
        job_data.get("generationCredits"),
        minimum_credits=0,
    )
    clip_retention_days = _parse_retention_days(job_data.get("clipRetentionDays"))
    smart_cleanup_enabled = bool(job_data.get("smartCleanupEnabled"))
    workspace_team_id = job_data.get("workspaceTeamId")
    billing_owner_user_id = job_data.get("billingOwnerUserId") or user_id
    charge_source = str(job_data.get("chargeSource") or "owner_wallet")
    workspace_role = str(job_data.get("workspaceRole") or "owner")
    smart_cleanup_summary = {
        "enabled": smart_cleanup_enabled,
        "profile": "balanced",
        "stopwords_removed": 0,
        "silence_seconds_removed": 0.0,
        "original_duration_seconds": 0.0,
        "output_duration_seconds": 0.0,
        "requested_window_start": 0.0,
        "requested_window_end": 0.0,
        "effective_window_start": 0.0,
        "effective_window_end": 0.0,
        "dropped_partial_words": 0,
    }

    # -- Per-clip working directory for isolation -------------------------
    work_dir = create_work_dir(f"clip_{clip_id}")

    generator = ClipGenerator(work_dir=work_dir)
    source_downloader = VideoDownloader(work_dir=work_dir)

    storage_path: str | None = None
    uploaded_storage_path: str | None = None
    source_video_strategy = "reused_existing"
    source_video_wait_seconds = 0.0
    source_video_download_seconds = 0.0

    try:
        _update_clip_job_progress(job_id, 0, "starting")

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
                "Insufficient credits for clip generation before processing starts: "
                f"required={generation_credits}, available={available}"
            )

        _update_clip_job_progress(job_id, 5, "loading_clip")

        # Get clip and video details
        clip_resp = (
            supabase.table("clips")
            .select("*, videos(*)")
            .eq("id", clip_id)
            .single()
            .execute()
        )
        assert_response_ok(clip_resp, f"Failed to load clip {clip_id}")
        clip = clip_resp.data
        if not clip:
            raise Exception(f"Clip {clip_id} not found")

        if not _is_latest_generate_job_for_clip(job_id=job_id, clip_id=clip_id):
            logger.info(
                "[%s] Skipping generation for clip %s because a newer job is active",
                job_id,
                clip_id,
            )
            _best_effort_mark_superseded(job_id=job_id, clip_id=clip_id)
            return

        # Update clip status
        clip_status_resp = supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()
        assert_response_ok(clip_status_resp, f"Failed to mark clip {clip_id} generating")

        _update_clip_job_progress(job_id, 12, "loading_layout")
        layout_selection = resolve_effective_layout_id(
            user_id=user_id,
            workspace_team_id=workspace_team_id,
            workspace_role=workspace_role,
            job_id=job_id,
            logger=logger,
            requested_layout_id=job_data.get("layoutId"),
            clip_layout_id=clip.get("layout_id"),
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
        output_quality = layout_overrides.output_quality

        # Unpack nested style dicts (layout values override task defaults)
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

        start_time = float(clip["start_time"])
        end_time = float(clip["end_time"])

        # -- Resolve source video path and dimensions -----------------------
        _update_clip_job_progress(job_id, 20, "preparing_source_video")
        waiting_stage_emitted = False

        def _emit_waiting_for_source(_elapsed_seconds: float):
            nonlocal waiting_stage_emitted
            if waiting_stage_emitted:
                return
            waiting_stage_emitted = True
            _update_clip_job_progress(job_id, 30, "waiting_for_source_video_download")

        def _emit_downloading_source():
            _update_clip_job_progress(job_id, 30, "downloading_source_video")

        source_resolution = resolve_source_video(
            video_id=clip["video_id"],
            source_url=clip["videos"].get("url"),
            initial_raw_video_path=clip["videos"].get("raw_video_path"),
            downloader=source_downloader,
            load_video_row=_load_video_source_row,
            persist_raw_video_path=_persist_video_raw_path,
            logger=logger,
            job_id=job_id,
            on_wait_for_download=_emit_waiting_for_source,
            on_download_start=_emit_downloading_source,
        )
        video_file = source_resolution.video_path
        src_w = int(source_resolution.width)
        src_h = int(source_resolution.height)
        source_video_strategy = str(source_resolution.strategy)
        source_video_wait_seconds = float(source_resolution.wait_seconds)
        source_video_download_seconds = float(source_resolution.download_seconds)

        logger.info(
            "[%s] Source video resolved for %s via %s (wait=%.2fs, download=%.2fs)",
            job_id,
            clip["video_id"],
            source_video_strategy,
            source_video_wait_seconds,
            source_video_download_seconds,
        )

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

        _update_clip_job_progress(job_id, 40, "preparing_captions")

        # -- Whisper re-transcription for word-level caption styles -----------
        # Prefer clip-level Whisper transcript (from a previous generation),
        # fall back to the video-level transcript (YouTube or Whisper).
        transcript = clip.get("transcript") or clip["videos"].get("transcript")
        caption_style = resolve_caption_style_mode(cap_cfg)
        should_retranscribe_for_captions = needs_whisper_retranscription(
            transcript, caption_style
        )
        has_word_timing_for_window = transcript_has_word_timing_in_window(
            transcript,
            start_time=start_time,
            end_time=end_time,
            minimum_words=1,
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
            logger.info(
                "[%s] Re-transcribing clip segment with Whisper (%s)",
                job_id,
                retranscribe_reason,
            )
            _update_clip_job_progress(job_id, 42, "retranscribing_with_whisper")
            video_duration = clip["videos"].get("duration_seconds") or (end_time + 10)
            transcript = transcribe_clip_window_with_whisper(
                media_path=video_file,
                work_dir=work_dir,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                video_duration_seconds=float(video_duration),
            )
            # Store the Whisper transcript on the clip row for reuse
            try:
                supabase.table("clips").update(
                    {"transcript": transcript}
                ).eq("id", clip_id).execute()
                logger.info("[%s] Stored Whisper transcript on clip %s", job_id, clip_id)
            except Exception:
                logger.warning(
                    "[%s] Failed to store Whisper transcript in DB; "
                    "continuing with ephemeral transcript",
                    job_id,
                    exc_info=True,
                )
            _update_clip_job_progress(job_id, 50, "preparing_captions")
        if smart_cleanup_enabled and not transcript_has_word_timing(transcript):
            raise RuntimeError(
                "Smart Cleanup requires Whisper word-level timings, but no usable words were returned."
            )

        if smart_cleanup_enabled:
            _update_clip_job_progress(job_id, 54, "applying_smart_cleanup")
            cleanup_result = apply_balanced_smart_cleanup(
                transcript=transcript,
                video_path=video_file,
                clip_id=clip_id,
                work_dir=work_dir,
                start_time=start_time,
                end_time=end_time,
            )
            video_file = cleanup_result["video_path"]
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
            _update_clip_job_progress(job_id, 58, "preparing_captions")

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

        # -- Generate clip ------------------------------------------------
        _update_clip_job_progress(job_id, 60, "rendering_clip")
        logger.info("[%s] Generating clip %s ...", job_id, clip_id)
        result = generator.generate(
            video_path=video_file,
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            title=clip["title"],
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
            # misc
            blur_strength=blur_strength,
            output_quality=output_quality,
        )

        if not _is_latest_generate_job_for_clip(job_id=job_id, clip_id=clip_id):
            logger.info(
                "[%s] Skipping upload for clip %s because a newer job is active",
                job_id,
                clip_id,
            )
            _best_effort_mark_superseded(job_id=job_id, clip_id=clip_id)
            return

        # Upload to Supabase Storage
        storage_path = f"clips/{clip_id}.mp4"

        _update_clip_job_progress(job_id, 80, "uploading_clip")
        logger.info("[%s] Uploading clip to storage ...", job_id)

        _upload_clip_with_replace(
            local_clip_path=result["clip_path"],
            storage_path=storage_path,
            job_id=job_id,
        )
        uploaded_storage_path = storage_path

        if not _is_latest_generate_job_for_clip(job_id=job_id, clip_id=clip_id):
            logger.info(
                "[%s] Uploaded output for stale generation job on clip %s; skipping finalize",
                job_id,
                clip_id,
            )
            _best_effort_mark_superseded(job_id=job_id, clip_id=clip_id)
            return

        # Charge credits atomically before finalizing clip completion status.
        _update_clip_job_progress(job_id, 90, "charging_credits")
        charge_clip_generation_credits(
            user_id=user_id,
            amount=generation_credits,
            description=f'Clip generation: {clip["title"][:50]}',
            video_id=clip["video_id"],
            clip_id=clip_id,
            charge_source=charge_source,
            team_id=workspace_team_id,
            billing_owner_user_id=billing_owner_user_id,
            actor_user_id=user_id,
            job_id=job_id,
            usage_metadata={
                "units_generated": 1,
                "smart_cleanup_enabled": smart_cleanup_enabled,
                "clip_duration_seconds": max(0.0, end_time - start_time),
                "workspace_role": workspace_role,
                "generation_flow": "suggested",
            },
        )

        # Update clip record
        _update_clip_job_progress(job_id, 97, "finalizing")
        clip_update = {
            "status": "completed",
            "storage_path": storage_path,
            "thumbnail_path": None,
            "file_size_bytes": result["file_size"],
            "asset_expires_at": _asset_expires_at_iso(clip_retention_days),
            "asset_expired_at": None,
        }
        if layout_should_persist and layout_id:
            clip_update["layout_id"] = layout_id
        clip_update_resp = (
            supabase.table("clips").update(clip_update).eq("id", clip_id).execute()
        )
        assert_response_ok(clip_update_resp, f"Failed to finalize clip {clip_id}")

        update_job_status(
            job_id,
            "completed",
            100,
            result_data={
                "stage": "completed",
                "storage_path": storage_path,
                "file_size": result["file_size"],
                "source_video_strategy": source_video_strategy,
                "source_video_wait_seconds": round(source_video_wait_seconds, 3),
                "source_video_download_seconds": round(
                    source_video_download_seconds, 3
                ),
                "smart_cleanup": smart_cleanup_summary,
            },
        )
        logger.info("[%s] Clip generation completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        _best_effort_cleanup_uploaded_artifacts(
            job_id=job_id,
            clip_id=clip_id,
            storage_path=uploaded_storage_path,
        )
        _best_effort_mark_failed(job_id=job_id, clip_id=clip_id, error_msg=error_msg)

        raise

    finally:
        # Remove the entire per-clip working directory (all intermediates)
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

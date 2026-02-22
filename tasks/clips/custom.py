"""Custom clip task -- download, transcript, and generate in one shot.

Combines the download + transcript logic from ``analyze_video_task`` with the
layout-loading and rendering logic from ``generate_clip_task`` so the user can
go straight from a URL + timestamps to a finished clip.
"""

import logging
import os
import shutil
import traceback

from config import CREDIT_COST_CLIP_GENERATION
from services.clip_generator import ClipGenerator, compute_video_position
from services.clips.constants import canvas_size_for_aspect_ratio
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.captions import build_caption_ass
from tasks.clips.helpers.layout import (
    load_layout_overrides,
    maybe_download_layout_background_image,
    resolve_effective_layout_id,
)
from tasks.clips.helpers.media import probe_video_size
from tasks.models.jobs import CustomClipJob
from tasks.models.layout import merge_layout_configs
from tasks.videos.transcript import (
    transcribe_clip_window_with_whisper,
)
from utils.workdirs import create_work_dir
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    get_credit_balance,
    has_sufficient_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)

_VIDEO_UPLOAD_OPTIONS = {"content-type": "video/mp4", "cache-control": "3600"}


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


def _update_clip_job_progress(job_id: str, progress: int, stage: str):
    """Persist custom clip-generation progress plus machine-readable stage."""
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


def _best_effort_mark_failed(
    *,
    job_id: str,
    clip_id: str,
    video_id: str,
    error_msg: str,
):
    try:
        update_job_status(job_id, "failed", 0, error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to update job failed status: %s", job_id, exc)

    try:
        fail_clip_resp = (
            supabase.table("clips")
            .update({"status": "failed", "error_message": error_msg})
            .eq("id", clip_id)
            .execute()
        )
        assert_response_ok(fail_clip_resp, f"Failed to mark clip {clip_id} failed")
    except Exception as exc:
        logger.warning("[%s] Failed to mark clip %s failed: %s", job_id, clip_id, exc)

    try:
        update_video_status(video_id, "failed", error_message=error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to mark video %s failed: %s", job_id, video_id, exc)


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
    generation_credits = int(job_data.get("generationCredits") or CREDIT_COST_CLIP_GENERATION)

    # -- Per-job working directory ------------------------------------------
    work_dir = create_work_dir(f"custom_{clip_id}")

    downloader = VideoDownloader(work_dir=work_dir)
    audio_path: str | None = None
    uploaded_storage_path: str | None = None

    try:
        _update_clip_job_progress(job_id, 0, "starting")

        if generation_credits > 0 and not has_sufficient_credits(
            user_id=user_id,
            amount=generation_credits,
        ):
            available = get_credit_balance(user_id)
            raise RuntimeError(
                "Insufficient credits for custom clip generation before processing starts: "
                f"required={generation_credits}, available={available}"
            )

        update_video_status(video_id, "downloading")
        clip_status_resp = supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()
        assert_response_ok(clip_status_resp, f"Failed to mark clip {clip_id} generating")

        # 1. Download video ----------------------------------------------------
        _update_clip_job_progress(job_id, 12, "downloading_video")
        logger.info("[%s] Downloading video: %s", job_id, url)
        video_data = downloader.download(url, video_id)
        video_path = video_data["path"]
        _update_clip_job_progress(job_id, 20, "downloading_video")

        duration_seconds = int(video_data["duration"])

        # Update video row with metadata
        update_video_status(
            video_id,
            "analyzing",
            title=video_data["title"],
            duration_seconds=duration_seconds,
            thumbnail_url=video_data["thumbnail"],
            platform=video_data["platform"],
            external_id=video_data.get("external_id"),
        )

        # 2. Get transcript ----------------------------------------------------
        transcript = None
        partial_transcript = False

        # Reuse stored transcript if the video already has one.
        existing_video_resp = (
            supabase.table("videos")
            .select("transcript")
            .eq("id", video_id)
            .single()
            .execute()
        )
        assert_response_ok(
            existing_video_resp,
            f"Failed to load existing transcript for {video_id}",
        )
        existing_video = existing_video_resp.data or {}
        existing_transcript = existing_video.get("transcript")
        if isinstance(existing_transcript, dict) and existing_transcript.get("segments"):
            transcript = existing_transcript
            logger.info("[%s] Reusing transcript already stored on video", job_id)

        if not transcript and video_data["platform"] == "youtube" and video_data.get(
            "external_id"
        ):
            _update_clip_job_progress(job_id, 30, "fetching_source_captions")
            logger.info("[%s] Attempting to get YouTube transcript ...", job_id)
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                logger.info("[%s] Got transcript from YouTube", job_id)

        if not transcript:
            _update_clip_job_progress(job_id, 30, "extracting_audio")
            logger.info("[%s] Extracting audio for Whisper transcription ...", job_id)
            audio_path = downloader.extract_audio(video_path)
            _update_clip_job_progress(job_id, 38, "extracting_audio")

            _update_clip_job_progress(job_id, 45, "transcribing_audio")
            transcript = transcribe_clip_window_with_whisper(
                audio_path=audio_path,
                work_dir=work_dir,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                video_duration_seconds=duration_seconds,
            )
            partial_transcript = True

        _update_clip_job_progress(job_id, 52, "loading_layout")

        # Save transcript on video row only when it represents the full video.
        video_update = {"status": "completed"}
        if not partial_transcript:
            video_update["transcript"] = transcript

        save_video_resp = (
            supabase.table("videos").update(video_update).eq("id", video_id).execute()
        )
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        # 3. Load layout -------------------------------------------------------
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
            job_id=job_id,
            logger=logger,
            requested_layout_id=job_data.get("layoutId"),
            clip_layout_id=clip_layout_id,
        )
        layout_id = layout_selection.layout_id
        layout_should_persist = layout_selection.should_persist_to_clip
        layout_overrides = load_layout_overrides(
            user_id=user_id,
            layout_id=layout_id,
            job_id=job_id,
            logger=logger,
        )
        bg_style = layout_overrides.bg_style
        bg_color = layout_overrides.bg_color
        blur_strength = layout_overrides.blur_strength
        output_quality = layout_overrides.output_quality

        vid_cfg, title_cfg, cap_cfg = merge_layout_configs(
            layout_overrides.layout_video,
            layout_overrides.layout_title,
            layout_overrides.layout_captions,
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

        _update_clip_job_progress(job_id, 65, "preparing_captions")
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
        _update_clip_job_progress(job_id, 75, "rendering_clip")
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
            # misc
            blur_strength=blur_strength,
            output_quality=output_quality,
        )

        # 7. Upload to Supabase Storage ----------------------------------------
        storage_path = f"clips/{clip_id}.mp4"

        _update_clip_job_progress(job_id, 84, "uploading_clip")
        logger.info("[%s] Uploading clip to storage ...", job_id)

        _upload_clip_with_replace(
            local_clip_path=result["clip_path"],
            storage_path=storage_path,
            job_id=job_id,
        )
        uploaded_storage_path = storage_path

        # 8. Charge credits before finalizing completion state -----------------
        _update_clip_job_progress(job_id, 90, "charging_credits")
        charge_clip_generation_credits(
            user_id=user_id,
            amount=generation_credits,
            description=f"Custom clip: {title[:50]}",
            video_id=video_id,
            clip_id=clip_id,
        )

        # 9. Update clip record ------------------------------------------------
        _update_clip_job_progress(job_id, 97, "finalizing")
        clip_update = {
            "status": "completed",
            "storage_path": storage_path,
            "thumbnail_path": None,
            "file_size_bytes": result["file_size"],
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
            },
        )
        logger.info("[%s] Custom clip completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating custom clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        _best_effort_cleanup_uploaded_artifacts(
            job_id=job_id,
            clip_id=clip_id,
            storage_path=uploaded_storage_path,
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

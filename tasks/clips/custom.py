"""Custom clip task -- download, transcript, and generate in one shot.

Combines the download + transcript logic from ``analyze_video_task`` with the
layout-loading and rendering logic from ``generate_clip_task`` so the user can
go straight from a URL + timestamps to a finished clip.
"""

import logging
import os
import shutil
import traceback

from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR
from services.clip_generator import ClipGenerator, compute_video_position
from services.clips.constants import canvas_size_for_aspect_ratio
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.captions import build_caption_ass
from tasks.clips.helpers.layout import (
    load_layout_overrides,
    maybe_download_layout_background_image,
)
from tasks.clips.helpers.media import probe_video_size
from tasks.models.jobs import CustomClipJob
from tasks.models.layout import merge_layout_configs
from tasks.videos.transcript import (
    transcribe_clip_window_with_whisper,
)
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)


def _best_effort_cleanup_uploaded_artifacts(
    *,
    job_id: str,
    clip_id: str,
    storage_path: str | None,
    thumbnail_path: str | None,
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

    if thumbnail_path:
        try:
            supabase.storage.from_("thumbnails").remove([thumbnail_path])
        except Exception as exc:
            logger.warning(
                "[%s] Failed to delete uploaded thumbnail artifact %s: %s",
                job_id,
                thumbnail_path,
                exc,
            )

    if storage_path or thumbnail_path:
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
        layoutId (optional)
    """
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]
    url = job_data["url"]
    start_time = float(job_data["startTime"])
    end_time = float(job_data["endTime"])
    title = job_data["title"]

    # -- Per-job working directory ------------------------------------------
    work_dir = os.path.join(TEMP_DIR, f"custom_{clip_id}")
    os.makedirs(work_dir, exist_ok=True)

    downloader = VideoDownloader(work_dir=work_dir)
    audio_path: str | None = None
    uploaded_storage_path: str | None = None
    uploaded_thumbnail_path: str | None = None

    try:
        update_job_status(job_id, "processing", 0)
        update_video_status(video_id, "downloading")
        clip_status_resp = supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()
        assert_response_ok(clip_status_resp, f"Failed to mark clip {clip_id} generating")

        # 1. Download video ----------------------------------------------------
        logger.info("[%s] Downloading video: %s", job_id, url)
        video_data = downloader.download(url, video_id)
        video_path = video_data["path"]
        update_job_status(job_id, "processing", 20)

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
            logger.info("[%s] Attempting to get YouTube transcript ...", job_id)
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                logger.info("[%s] Got transcript from YouTube", job_id)

        if not transcript:
            logger.info("[%s] Extracting audio for Whisper transcription ...", job_id)
            audio_path = downloader.extract_audio(video_path)
            update_job_status(job_id, "processing", 30)

            transcript = transcribe_clip_window_with_whisper(
                audio_path=audio_path,
                work_dir=work_dir,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                video_duration_seconds=duration_seconds,
            )
            partial_transcript = True

        update_job_status(job_id, "processing", 40)

        # Save transcript on video row only when it represents the full video.
        video_update = {"status": "completed"}
        if not partial_transcript:
            video_update["transcript"] = transcript

        save_video_resp = (
            supabase.table("videos").update(video_update).eq("id", video_id).execute()
        )
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        # 3. Load layout -------------------------------------------------------
        layout_id = job_data.get("layoutId")
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
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            video_scale_mode=video_scale_mode,
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

        update_job_status(job_id, "processing", 50)

        # 6. Generate clip -----------------------------------------------------
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
            # captions
            caption_ass_path=caption_ass_path,
            # misc
            blur_strength=blur_strength,
            output_quality=output_quality,
        )

        update_job_status(job_id, "processing", 90)

        # 7. Upload to Supabase Storage ----------------------------------------
        storage_path = f"clips/{clip_id}.mp4"
        thumbnail_path = f"thumbnails/{clip_id}.jpg"

        logger.info("[%s] Uploading clip to storage ...", job_id)

        with open(result["clip_path"], "rb") as f:
            supabase.storage.from_("generated-clips").upload(storage_path, f)
        uploaded_storage_path = storage_path

        with open(result["thumbnail_path"], "rb") as f:
            supabase.storage.from_("thumbnails").upload(thumbnail_path, f)
        uploaded_thumbnail_path = thumbnail_path

        # 8. Charge credits before finalizing completion state -----------------
        charge_clip_generation_credits(
            user_id=user_id,
            amount=CREDIT_COST_CLIP_GENERATION,
            description=f"Custom clip: {title[:50]}",
            video_id=video_id,
            clip_id=clip_id,
        )

        # 9. Update clip record ------------------------------------------------
        clip_update = {
            "status": "completed",
            "storage_path": storage_path,
            "thumbnail_path": thumbnail_path,
            "file_size_bytes": result["file_size"],
        }
        if layout_id:
            clip_update["layout_id"] = layout_id
        clip_update_resp = (
            supabase.table("clips").update(clip_update).eq("id", clip_id).execute()
        )
        assert_response_ok(clip_update_resp, f"Failed to finalize clip {clip_id}")

        update_job_status(job_id, "completed", 100, result_data={
            "storage_path": storage_path,
            "thumbnail_path": thumbnail_path,
            "file_size": result["file_size"],
        })
        logger.info("[%s] Custom clip completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating custom clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        _best_effort_cleanup_uploaded_artifacts(
            job_id=job_id,
            clip_id=clip_id,
            storage_path=uploaded_storage_path,
            thumbnail_path=uploaded_thumbnail_path,
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

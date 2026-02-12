"""Custom clip task -- download, transcript, and generate in one shot.

Combines the download + transcript logic from ``analyze_video_task`` with the
layout-loading and rendering logic from ``generate_clip_task`` so the user can
go straight from a URL + timestamps to a finished clip.
"""

import logging
import os
import shutil
import traceback

import ffmpeg as ffmpeg_lib

from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR
from services.caption_renderer import extract_clip_segments, render_ass
from services.clip_generator import ClipGenerator, compute_video_position
from services.transcriber import Transcriber
from services.video_downloader import VideoDownloader
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)

# Defaults that match the layout JSON schema
_DEFAULT_VIDEO = {"widthPct": 100, "positionY": "middle"}
_DEFAULT_TITLE = {
    "show": True,
    "fontSize": 48,
    "fontColor": "white",
    "fontFamily": "",
    "align": "left",
    "strokeWidth": 0,
    "strokeColor": "black",
    "barEnabled": True,
    "barColor": "black@0.5",
    "paddingX": 16,
    "positionY": "above_video",
}
_DEFAULT_CAPTIONS = {
    "show": False,
    "style": "animated",
    "fontSize": 42,
    "fontColor": "white",
    "fontFamily": "",
    "highlightColor": "#FFD700",
    "position": "bottom",
    "maxWordsPerLine": 5,
}


def custom_clip_task(job_data: dict):
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
        if video_data["platform"] == "youtube" and video_data.get("external_id"):
            logger.info("[%s] Attempting to get YouTube transcript …", job_id)
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                logger.info("[%s] Got transcript from YouTube", job_id)

        if not transcript:
            logger.info("[%s] Extracting audio for Whisper transcription …", job_id)
            audio_path = downloader.extract_audio(video_path)
            update_job_status(job_id, "processing", 30)

            logger.info("[%s] Transcribing with Whisper …", job_id)
            transcriber = Transcriber()
            transcript = transcriber.transcribe(audio_path)
            transcript["source"] = "whisper"

        update_job_status(job_id, "processing", 40)

        # Save transcript on video row
        save_video_resp = supabase.table("videos").update(
            {"status": "completed", "transcript": transcript}
        ).eq("id", video_id).execute()
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        # 3. Load layout -------------------------------------------------------
        bg_style = "blur"
        bg_color = "#000000"
        bg_image_storage_path: str | None = None
        layout_video: dict = {}
        layout_title: dict = {}
        layout_captions: dict = {}
        blur_strength = 20
        output_quality = "medium"

        layout_id = job_data.get("layoutId")
        if layout_id:
            logger.info("[%s] Loading layout %s …", job_id, layout_id)
            layout_resp = (
                supabase.table("layouts")
                .select("*")
                .eq("id", layout_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            assert_response_ok(layout_resp, f"Failed to load layout {layout_id}")
            layout = layout_resp.data
            if layout:
                bg_style = layout.get("background_style", "blur")
                bg_color = layout.get("background_color") or "#000000"
                bg_image_storage_path = layout.get("background_image_path")
                blur_strength = layout.get("background_blur_strength") or 20
                output_quality = layout.get("output_quality") or "medium"
                layout_video = layout.get("video") or {}
                layout_title = layout.get("title") or {}
                layout_captions = layout.get("captions") or {}
            else:
                logger.warning(
                    "[%s] Layout %s not found, using defaults", job_id, layout_id
                )

        vid_cfg = {**_DEFAULT_VIDEO, **layout_video}
        title_cfg = {**_DEFAULT_TITLE, **layout_title}
        cap_cfg = {**_DEFAULT_CAPTIONS, **layout_captions}

        # Download background image from Supabase storage if needed
        bg_image_path: str | None = None
        if bg_style == "image" and bg_image_storage_path:
            bg_image_path = os.path.join(work_dir, "bg_image.jpg")
            try:
                file_bytes = supabase.storage.from_("layouts").download(
                    bg_image_storage_path
                )
                with open(bg_image_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(
                    "[%s] Downloaded background image from storage: %s",
                    job_id,
                    bg_image_storage_path,
                )
            except Exception as dl_err:
                logger.warning(
                    "[%s] Could not download background image, falling back to blur: %s",
                    job_id,
                    dl_err,
                )
                bg_style = "blur"
                bg_image_path = None

        # 4. Compute video position for caption placement ----------------------
        probe = ffmpeg_lib.probe(video_path)
        v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        src_w = int(v_stream["width"])
        src_h = int(v_stream["height"])
        _, vid_h, _, vid_y = compute_video_position(
            src_w, src_h, vid_cfg["widthPct"], vid_cfg["positionY"]
        )

        # 5. Build caption ASS file if requested -------------------------------
        caption_ass_path: str | None = None

        if cap_cfg["show"]:
            if transcript and transcript.get("segments"):
                logger.info("[%s] Building %s captions …", job_id, cap_cfg["style"])
                segments = extract_clip_segments(transcript, start_time, end_time)

                if segments:
                    caption_ass_path = render_ass(
                        segments,
                        style=cap_cfg["style"],
                        font_size=cap_cfg["fontSize"],
                        font_color=cap_cfg["fontColor"],
                        font_family=cap_cfg["fontFamily"],
                        highlight_color=cap_cfg["highlightColor"],
                        position=cap_cfg["position"],
                        max_words_per_line=cap_cfg["maxWordsPerLine"],
                        vid_y=vid_y,
                        vid_h=vid_h,
                        output_path=os.path.join(work_dir, f"{clip_id}.ass"),
                    )
            else:
                logger.warning(
                    "[%s] Captions requested but no transcript available", job_id
                )

        update_job_status(job_id, "processing", 50)

        # 6. Generate clip -----------------------------------------------------
        generator = ClipGenerator(work_dir=work_dir)
        logger.info("[%s] Generating clip %s …", job_id, clip_id)

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

        logger.info("[%s] Uploading clip to storage …", job_id)

        with open(result["clip_path"], "rb") as f:
            supabase.storage.from_("generated-clips").upload(storage_path, f)

        with open(result["thumbnail_path"], "rb") as f:
            supabase.storage.from_("thumbnails").upload(thumbnail_path, f)

        # 8. Update clip record ------------------------------------------------
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

        # 9. Charge credits atomically -----------------------------------------
        charge_clip_generation_credits(
            user_id=user_id,
            amount=CREDIT_COST_CLIP_GENERATION,
            description=f"Custom clip: {title[:50]}",
            video_id=video_id,
            clip_id=clip_id,
        )

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

        update_job_status(job_id, "failed", 0, error_msg)
        fail_clip_resp = supabase.table("clips").update(
            {"status": "failed", "error_message": error_msg}
        ).eq("id", clip_id).execute()
        assert_response_ok(fail_clip_resp, f"Failed to mark clip {clip_id} failed")
        update_video_status(video_id, "failed", error_message=error_msg)

        raise

    finally:
        # Clean up audio if extracted
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        # Remove the entire per-clip working directory
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

import logging
import os
import shutil
import traceback

import ffmpeg as ffmpeg_lib

from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR
from services.caption_renderer import extract_clip_segments, render_ass
from services.clip_generator import ClipGenerator, compute_video_position
from services.video_downloader import VideoDownloader
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    supabase,
    update_job_status,
)

logger = logging.getLogger(__name__)

# Defaults that match the API Pydantic models
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


def _probe_video_size(video_path: str) -> tuple[int, int]:
    probe = ffmpeg_lib.probe(video_path)
    v_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if v_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")
    return int(v_stream["width"]), int(v_stream["height"])


def generate_clip_task(job_data: dict):
    """Generate final clip with overlay.

    Each invocation creates a per-clip working directory so multiple
    clip-generation workers can run concurrently without collisions.
    The entire directory is removed in the ``finally`` block.
    """
    job_id = job_data["jobId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]

    # -- Load layout from Supabase if provided --------------------------------
    bg_style = "blur"
    bg_color = "#000000"
    bg_image_storage_path: str | None = None  # path inside Supabase storage
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
            logger.warning("[%s] Layout %s not found, using defaults", job_id, layout_id)

    # Unpack nested style dicts (layout values override built-in defaults)
    vid_cfg = {**_DEFAULT_VIDEO, **layout_video}
    title_cfg = {**_DEFAULT_TITLE, **layout_title}
    cap_cfg = {**_DEFAULT_CAPTIONS, **layout_captions}

    # -- Per-clip working directory for isolation -------------------------
    work_dir = os.path.join(TEMP_DIR, f"clip_{clip_id}")
    os.makedirs(work_dir, exist_ok=True)

    generator = ClipGenerator(work_dir=work_dir)

    # Download background image from Supabase storage if needed
    bg_image_path: str | None = None
    if bg_style == "image" and bg_image_storage_path:
        bg_image_path = os.path.join(work_dir, "bg_image.jpg")
        try:
            # Download from Supabase storage bucket "layouts"
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

    try:
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

        update_job_status(job_id, "processing", 0)

        # Update clip status
        clip_status_resp = supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()
        assert_response_ok(clip_status_resp, f"Failed to mark clip {clip_id} generating")

        start_time = float(clip["start_time"])
        end_time = float(clip["end_time"])

        # -- Resolve source video path and dimensions -----------------------
        video_file = clip["videos"].get("raw_video_path")
        src_w: int | None = None
        src_h: int | None = None

        if video_file and os.path.isfile(video_file):
            try:
                src_w, src_h = _probe_video_size(video_file)
            except Exception as exc:
                logger.warning(
                    "[%s] Existing raw video is not probeable (%s): %s",
                    job_id,
                    video_file,
                    exc,
                )
                video_file = None
        else:
            video_file = None

        if not video_file:
            source_url = clip["videos"].get("url")
            if not source_url:
                raise RuntimeError("Missing source URL and raw_video_path for clip generation")

            logger.info("[%s] Re-downloading source video for clip %s", job_id, clip_id)
            downloader = VideoDownloader(work_dir=work_dir)
            downloaded = downloader.download(source_url, clip["video_id"])
            video_file = downloaded["path"]
            src_w, src_h = _probe_video_size(video_file)

        assert src_w is not None and src_h is not None
        _, vid_h, _, vid_y = compute_video_position(
            src_w, src_h, vid_cfg["widthPct"], vid_cfg["positionY"]
        )

        # -- Build caption ASS file if requested --------------------------
        caption_ass_path: str | None = None

        if cap_cfg["show"]:
            transcript = clip["videos"].get("transcript")
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

        # -- Generate clip ------------------------------------------------
        logger.info("[%s] Generating clip %s …", job_id, clip_id)
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

        # Upload to Supabase Storage
        storage_path = f"clips/{clip_id}.mp4"
        thumbnail_path = f"thumbnails/{clip_id}.jpg"

        logger.info("[%s] Uploading clip to storage …", job_id)

        with open(result["clip_path"], "rb") as f:
            supabase.storage.from_("generated-clips").upload(storage_path, f)

        with open(result["thumbnail_path"], "rb") as f:
            supabase.storage.from_("thumbnails").upload(thumbnail_path, f)

        # Update clip record
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

        # Charge credits atomically (fails on insufficient balance).
        charge_clip_generation_credits(
            user_id=user_id,
            amount=CREDIT_COST_CLIP_GENERATION,
            description=f'Clip generation: {clip["title"][:50]}',
            video_id=clip["video_id"],
            clip_id=clip_id,
        )

        update_job_status(job_id, "completed", 100, result_data={
            "storage_path": storage_path,
            "thumbnail_path": thumbnail_path,
            "file_size": result["file_size"],
        })
        logger.info("[%s] Clip generation completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        update_job_status(job_id, "failed", 0, error_msg)
        fail_resp = supabase.table("clips").update(
            {"status": "failed", "error_message": error_msg}
        ).eq("id", clip_id).execute()
        assert_response_ok(fail_resp, f"Failed to mark clip {clip_id} failed")

        raise

    finally:
        # Remove the entire per-clip working directory (all intermediates)
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

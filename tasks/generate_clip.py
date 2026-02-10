import logging
import os
import shutil
import traceback

import ffmpeg as ffmpeg_lib

from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR
from services.caption_renderer import extract_clip_segments, render_ass
from services.clip_generator import ClipGenerator, compute_video_position
from utils.supabase_client import supabase, update_job_status

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
        clip = (
            supabase.table("clips")
            .select("*, videos(*)")
            .eq("id", clip_id)
            .single()
            .execute()
            .data
        )

        # Check credits
        result = supabase.rpc(
            "has_credits",
            {"p_user_id": user_id, "p_amount": CREDIT_COST_CLIP_GENERATION},
        ).execute()

        if not result.data:
            raise Exception("Insufficient credits")

        update_job_status(job_id, "processing", 0)

        # Update clip status
        supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()

        start_time = float(clip["start_time"])
        end_time = float(clip["end_time"])

        # -- Compute video position for caption placement -------------------
        video_file = clip["videos"]["raw_video_path"]
        probe = ffmpeg_lib.probe(video_file)
        v_stream = next(
            s for s in probe["streams"] if s["codec_type"] == "video"
        )
        src_w = int(v_stream["width"])
        src_h = int(v_stream["height"])
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
            video_path=clip["videos"]["raw_video_path"],
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
        supabase.table("clips").update(
            {
                "status": "completed",
                "storage_path": storage_path,
                "thumbnail_path": thumbnail_path,
                "file_size_bytes": result["file_size"],
            }
        ).eq("id", clip_id).execute()

        # Charge credits
        supabase.rpc(
            "charge_credits",
            {
                "p_user_id": user_id,
                "p_amount": CREDIT_COST_CLIP_GENERATION,
                "p_type": "clip_generation",
                "p_description": f'Clip generation: {clip["title"][:50]}',
                "p_clip_id": clip_id,
            },
        ).execute()

        update_job_status(job_id, "completed", 100)
        logger.info("[%s] Clip generation completed: %s", job_id, clip_id)

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error generating clip: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        update_job_status(job_id, "failed", 0, error_msg)
        supabase.table("clips").update(
            {"status": "failed", "error_message": error_msg}
        ).eq("id", clip_id).execute()

        raise

    finally:
        # Remove the entire per-clip working directory (all intermediates)
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

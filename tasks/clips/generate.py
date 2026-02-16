import logging
import os
import shutil
import traceback

from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR
from services.caption_renderer import normalize_caption_style
from services.clip_generator import ClipGenerator, compute_video_position
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.captions import build_caption_ass
from tasks.clips.helpers.layout import (
    load_layout_overrides,
    maybe_download_layout_background_image,
)
from tasks.clips.helpers.media import probe_video_size
from tasks.models.jobs import GenerateClipJob
from tasks.models.layout import merge_layout_configs
from utils.supabase_client import (
    assert_response_ok,
    charge_clip_generation_credits,
    supabase,
    update_job_status,
)

logger = logging.getLogger(__name__)


def _best_effort_cleanup_uploaded_artifacts(
    *,
    job_id: str,
    clip_id: str,
    storage_path: str | None,
    thumbnail_path: str | None,
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


def generate_clip_task(job_data: GenerateClipJob):
    """Generate final clip with overlay.

    Each invocation creates a per-clip working directory so multiple
    clip-generation workers can run concurrently without collisions.
    The entire directory is removed in the ``finally`` block.
    """
    job_id = job_data["jobId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]

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

    # Unpack nested style dicts (layout values override task defaults)
    vid_cfg, title_cfg, cap_cfg = merge_layout_configs(
        layout_overrides.layout_video,
        layout_overrides.layout_title,
        layout_overrides.layout_captions,
    )
    normalized_caption_style = normalize_caption_style(cap_cfg.get("style"))

    # -- Per-clip working directory for isolation -------------------------
    work_dir = os.path.join(TEMP_DIR, f"clip_{clip_id}")
    os.makedirs(work_dir, exist_ok=True)

    generator = ClipGenerator(work_dir=work_dir)

    bg_style, bg_image_path = maybe_download_layout_background_image(
        bg_style=bg_style,
        bg_image_storage_path=layout_overrides.bg_image_storage_path,
        work_dir=work_dir,
        job_id=job_id,
        logger=logger,
    )

    storage_path: str | None = None
    thumbnail_path: str | None = None
    uploaded_storage_path: str | None = None
    uploaded_thumbnail_path: str | None = None

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
                src_w, src_h = probe_video_size(video_file)
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
            src_w, src_h = probe_video_size(video_file)

        assert src_w is not None and src_h is not None
        _, vid_h, _, vid_y = compute_video_position(
            src_w, src_h, vid_cfg["widthPct"], vid_cfg["positionY"]
        )

        caption_ass_path = build_caption_ass(
            job_id=job_id,
            clip_id=clip_id,
            transcript=clip["videos"].get("transcript"),
            cap_cfg=cap_cfg,
            normalized_caption_style=normalized_caption_style,
            start_time=start_time,
            end_time=end_time,
            vid_y=vid_y,
            vid_h=vid_h,
            work_dir=work_dir,
            logger=logger,
        )

        # -- Generate clip ------------------------------------------------
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

        logger.info("[%s] Uploading clip to storage ...", job_id)

        with open(result["clip_path"], "rb") as f:
            supabase.storage.from_("generated-clips").upload(storage_path, f)
        uploaded_storage_path = storage_path

        with open(result["thumbnail_path"], "rb") as f:
            supabase.storage.from_("thumbnails").upload(thumbnail_path, f)
        uploaded_thumbnail_path = thumbnail_path

        # Charge credits atomically before finalizing clip completion status.
        charge_clip_generation_credits(
            user_id=user_id,
            amount=CREDIT_COST_CLIP_GENERATION,
            description=f'Clip generation: {clip["title"][:50]}',
            video_id=clip["video_id"],
            clip_id=clip_id,
        )

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

        update_job_status(
            job_id,
            "completed",
            100,
            result_data={
                "storage_path": storage_path,
                "thumbnail_path": thumbnail_path,
                "file_size": result["file_size"],
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
            thumbnail_path=uploaded_thumbnail_path,
        )
        _best_effort_mark_failed(job_id=job_id, clip_id=clip_id, error_msg=error_msg)

        raise

    finally:
        # Remove the entire per-clip working directory (all intermediates)
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug("[%s] Removed work dir %s", job_id, work_dir)

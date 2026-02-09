import logging
import os

import ffmpeg
from config import TEMP_DIR

logger = logging.getLogger(__name__)


class ClipGenerator:
    def __init__(self, work_dir: str | None = None):
        self.temp_dir = work_dir or TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        video_path: str,
        clip_id: str,
        start_time: float,
        end_time: float,
        title: str,
        background_style: str = "blur",
    ) -> dict:
        """Generate final portrait clip with overlay.

        Returns dict with ``clip_path``, ``thumbnail_path``, ``file_size``
        and an ``intermediates`` list of temp files the caller may clean up.
        """
        intermediates: list[str] = []

        # 1. Extract clip segment
        raw_clip_path = os.path.join(self.temp_dir, f"{clip_id}_raw.mp4")
        self._extract_segment(video_path, start_time, end_time, raw_clip_path)
        intermediates.append(raw_clip_path)

        # 2. Create portrait background
        bg_clip_path = os.path.join(self.temp_dir, f"{clip_id}_bg.mp4")
        self._create_portrait_background(raw_clip_path, bg_clip_path, background_style)
        intermediates.append(bg_clip_path)

        # 3. Add title overlay (preserving audio from the raw clip)
        final_clip_path = os.path.join(self.temp_dir, f"{clip_id}_final.mp4")
        self._add_title_overlay(bg_clip_path, raw_clip_path, final_clip_path, title)

        # 4. Generate thumbnail
        thumbnail_path = os.path.join(self.temp_dir, f"{clip_id}_thumb.jpg")
        self._generate_thumbnail(final_clip_path, thumbnail_path)

        file_size = os.path.getsize(final_clip_path)

        return {
            "clip_path": final_clip_path,
            "thumbnail_path": thumbnail_path,
            "file_size": file_size,
            "intermediates": intermediates,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_segment(
        self, input_path: str, start: float, end: float, output_path: str
    ):
        """Extract video segment."""
        duration = end - start
        logger.info("Extracting segment %.2f–%.2f (%.2fs)", start, end, duration)

        (
            ffmpeg.input(input_path, ss=start, t=duration)
            .output(output_path, vcodec="libx264", acodec="aac")
            .overwrite_output()
            .run(quiet=True)
        )

    def _create_portrait_background(
        self, input_path: str, output_path: str, style: str
    ):
        """Create 9:16 portrait background."""
        logger.info("Creating portrait background (style=%s)", style)

        if style == "blur":
            bg_temp = output_path + ".bg.mp4"

            # Blurred background
            (
                ffmpeg.input(input_path)
                .filter("scale", 1080, 1920, force_original_aspect_ratio="increase")
                .filter("crop", 1080, 1920)
                .filter("boxblur", 20)
                .output(bg_temp)
                .overwrite_output()
                .run(quiet=True)
            )

            # Overlay original video centred
            bg = ffmpeg.input(bg_temp)
            video = ffmpeg.input(input_path)

            (
                ffmpeg.filter([bg, video], "overlay", "(W-w)/2", "(H-h)/2")
                .output(output_path, vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

            # Remove temporary blur file
            self._safe_remove(bg_temp)

        else:  # solid_color / gradient
            (
                ffmpeg.input(input_path)
                .filter("scale", 1080, 1920, force_original_aspect_ratio="decrease")
                .filter("pad", 1080, 1920, "(ow-iw)/2", "(oh-ih)/2", "black")
                .output(output_path, vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

    def _add_title_overlay(
        self, video_path: str, audio_path: str, output_path: str, title: str
    ):
        """Add title overlay using drawtext while preserving the original audio."""
        logger.info("Adding title overlay")

        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)

        video_with_text = video.drawtext(
            text=title,
            fontsize=48,
            fontcolor="white",
            x="(w-text_w)/2",
            y="80",
            shadowcolor="black@0.7",
            shadowx=3,
            shadowy=3,
            box=1,
            boxcolor="black@0.5",
            boxborderw=10,
        )

        (
            ffmpeg.output(
                video_with_text,
                audio.audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                audio_bitrate="192k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )

    def _generate_thumbnail(self, video_path: str, output_path: str):
        """Generate thumbnail at 1 second mark."""
        (
            ffmpeg.input(video_path, ss=1)
            .output(output_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )

    @staticmethod
    def _safe_remove(path: str):
        try:
            os.remove(path)
        except OSError:
            pass

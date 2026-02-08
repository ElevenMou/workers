import ffmpeg
import os
from PIL import Image, ImageDraw, ImageFont
from config import TEMP_DIR


class ClipGenerator:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def generate(
        self,
        video_path: str,
        clip_id: str,
        start_time: float,
        end_time: float,
        title: str,
        background_style: str = "blur",
    ) -> dict:
        """Generate final portrait clip with overlay"""

        # 1. Extract clip segment
        raw_clip_path = os.path.join(self.temp_dir, f"{clip_id}_raw.mp4")
        self._extract_segment(video_path, start_time, end_time, raw_clip_path)

        # 2. Create portrait background
        bg_clip_path = os.path.join(self.temp_dir, f"{clip_id}_bg.mp4")
        self._create_portrait_background(raw_clip_path, bg_clip_path, background_style)

        # 3. Add title overlay
        final_clip_path = os.path.join(self.temp_dir, f"{clip_id}_final.mp4")
        self._add_title_overlay(bg_clip_path, final_clip_path, title)

        # 4. Generate thumbnail
        thumbnail_path = os.path.join(self.temp_dir, f"{clip_id}_thumb.jpg")
        self._generate_thumbnail(final_clip_path, thumbnail_path)

        file_size = os.path.getsize(final_clip_path)

        return {
            "clip_path": final_clip_path,
            "thumbnail_path": thumbnail_path,
            "file_size": file_size,
        }

    def _extract_segment(
        self, input_path: str, start: float, end: float, output_path: str
    ):
        """Extract video segment"""
        duration = end - start

        (
            ffmpeg.input(input_path, ss=start, t=duration)
            .output(output_path, vcodec="libx264", acodec="aac")
            .overwrite_output()
            .run(quiet=True)
        )

    def _create_portrait_background(
        self, input_path: str, output_path: str, style: str
    ):
        """Create 9:16 portrait background"""

        if style == "blur":
            # Create blurred background + centered video
            (
                ffmpeg.input(input_path)
                .filter("scale", 1080, 1920, force_original_aspect_ratio="increase")
                .filter("crop", 1080, 1920)
                .filter("boxblur", 20)
                .output(output_path + ".bg.mp4")
                .overwrite_output()
                .run(quiet=True)
            )

            # Overlay original video centered
            bg = ffmpeg.input(output_path + ".bg.mp4")
            video = ffmpeg.input(input_path)

            (
                ffmpeg.filter([bg, video], "overlay", "(W-w)/2", "(H-h)/2")
                .output(output_path, vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

            os.remove(output_path + ".bg.mp4")

        else:  # solid_color or gradient
            # Simple centered video on black background
            (
                ffmpeg.input(input_path)
                .filter("scale", 1080, 1920, force_original_aspect_ratio="decrease")
                .filter("pad", 1080, 1920, "(ow-iw)/2", "(oh-ih)/2", "black")
                .output(output_path, vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

    def _add_title_overlay(self, input_path: str, output_path: str, title: str):
        """Add title overlay using drawtext"""

        (
            ffmpeg.input(input_path)
            .drawtext(
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
            .output(output_path, vcodec="libx264", acodec="copy")
            .overwrite_output()
            .run(quiet=True)
        )

    def _generate_thumbnail(self, video_path: str, output_path: str):
        """Generate thumbnail at 1 second mark"""
        (
            ffmpeg.input(video_path, ss=1)
            .output(output_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )

"""High-level clip generation orchestration."""

import os

from config import TEMP_DIR
from services.clips.constants import (
    DEFAULT_CANVAS_ASPECT_RATIO,
    DEFAULT_VIDEO_SCALE_MODE,
    QUALITY_PRESETS,
    canvas_size_for_aspect_ratio,
)
from services.clips.ffmpeg_ops import (
    add_overlays,
    create_portrait_background,
    extract_segment,
)
from services.clips.layout import compute_layout, compute_video_position, wrap_title
from services.clips.models import ClipGenerationResult


class ClipGenerator:
    def __init__(self, work_dir: str | None = None):
        self.temp_dir = work_dir or TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def generate(
        self,
        video_path: str,
        clip_id: str,
        start_time: float,
        end_time: float,
        title: str,
        background_style: str = "blur",
        background_color: str = "#000000",
        background_image_path: str | None = None,
        # -- video layout --
        video_width_pct: int = 100,
        video_position_y: str = "middle",
        video_custom_x: int | None = None,
        video_custom_y: int | None = None,
        video_custom_width: int | None = None,
        canvas_aspect_ratio: str = DEFAULT_CANVAS_ASPECT_RATIO,
        video_scale_mode: str = DEFAULT_VIDEO_SCALE_MODE,
        # -- title style --
        title_show: bool = True,
        title_font_size: int = 48,
        title_font_color: str = "#FFFFFF",
        title_font_family: str = "",
        title_align: str = "left",
        title_stroke_width: int = 0,
        title_stroke_color: str = "#000000",
        title_bar_enabled: bool = True,
        title_bar_color: str = "#000000",
        title_padding_x: int = 16,
        title_position_y: str = "above_video",
        title_custom_x: int | None = None,
        title_custom_y: int | None = None,
        title_custom_width: int | None = None,
        # -- captions --
        caption_ass_path: str | None = None,
        # -- misc --
        blur_strength: int = 20,
        output_quality: str = "medium",
    ) -> ClipGenerationResult:
        """Generate final clip with title and optional captions."""
        intermediates: list[str] = []
        qp = QUALITY_PRESETS.get(output_quality, QUALITY_PRESETS["medium"])
        canvas_w, _ = canvas_size_for_aspect_ratio(canvas_aspect_ratio)

        if str(title_position_y).strip().lower() == "custom":
            try:
                custom_title_width = (
                    int(title_custom_width) if title_custom_width is not None else canvas_w
                )
            except (TypeError, ValueError):
                custom_title_width = canvas_w
            base_title_width = max(2, min(canvas_w, custom_title_width))
        else:
            base_title_width = canvas_w
        max_text_w = max(120, base_title_width - 6 * max(0, int(title_padding_x)))
        title_lines = wrap_title(title, title_font_size, max_text_w)

        raw_clip_path = os.path.join(self.temp_dir, f"{clip_id}_raw.mp4")
        extract_segment(video_path, start_time, end_time, raw_clip_path, qp)
        intermediates.append(raw_clip_path)

        layout = compute_layout(
            raw_clip_path,
            video_width_pct,
            video_position_y,
            video_custom_x,
            video_custom_y,
            video_custom_width,
            title_font_size,
            title_padding_x,
            title_position_y,
            title_custom_x,
            title_custom_y,
            title_custom_width,
            canvas_aspect_ratio,
            video_scale_mode,
            len(title_lines),
        )

        bg_clip_path = os.path.join(self.temp_dir, f"{clip_id}_bg.mp4")
        create_portrait_background(
            raw_clip_path,
            bg_clip_path,
            background_style,
            layout["canvas_w"],
            layout["canvas_h"],
            layout["vid_w"],
            layout["vid_h"],
            layout["vid_x"],
            layout["vid_y"],
            blur_strength,
            video_scale_mode,
            qp,
            background_color=background_color,
            background_image_path=background_image_path,
        )
        intermediates.append(bg_clip_path)

        final_clip_path = os.path.join(self.temp_dir, f"{clip_id}_final.mp4")
        add_overlays(
            bg_clip_path,
            raw_clip_path,
            final_clip_path,
            title_lines=title_lines,
            title_show=title_show,
            title_font_size=title_font_size,
            title_font_color=title_font_color,
            title_font_family=title_font_family,
            title_align=title_align,
            title_stroke_width=title_stroke_width,
            title_stroke_color=title_stroke_color,
            title_bar_enabled=title_bar_enabled,
            title_bar_color=title_bar_color,
            title_bar_x=layout["title_bar_x"],
            title_bar_w=layout["title_bar_w"],
            title_padding_x=layout["title_padding_x"],
            title_bar_y=layout["title_bar_y"],
            title_text_y=layout["title_text_y"],
            title_bar_h=layout["title_bar_h"],
            caption_ass_path=caption_ass_path,
            qp=qp,
        )

        file_size = os.path.getsize(final_clip_path)
        return {
            "clip_path": final_clip_path,
            "file_size": file_size,
            "intermediates": intermediates,
        }


__all__ = ["ClipGenerator", "compute_video_position"]

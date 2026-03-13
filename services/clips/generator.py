"""High-level clip generation orchestration."""

from __future__ import annotations

import logging
import os
from typing import Any

import ffmpeg as ffmpeg_lib

from config import TEMP_DIR
from services.clips.constants import (
    DEFAULT_CANVAS_ASPECT_RATIO,
    DEFAULT_VIDEO_SCALE_MODE,
    QUALITY_PRESETS,
    canvas_size_for_aspect_ratio,
    intermediate_quality_preset,
)
from services.clips.ffmpeg_ops import (
    concat_intro_outro,
    compose_clip,
    safe_remove,
)
from services.clips.layout import compute_layout, compute_video_position, wrap_title
from services.clips.models import ClipGenerationResult

logger = logging.getLogger(__name__)


def _probe_video_resolution(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream in *path*."""
    probe = ffmpeg_lib.probe(path)
    v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(v_stream["width"]), int(v_stream["height"])


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
        # -- intro / outro / overlay --
        intro_cfg: dict[str, Any] | None = None,
        outro_cfg: dict[str, Any] | None = None,
        overlay_cfg: dict[str, Any] | None = None,
        intro_file_path: str | None = None,
        outro_file_path: str | None = None,
        overlay_file_path: str | None = None,
        # -- reframe (speaker tracking) --
        reframe_enabled: bool = False,
        reframe_smoothing: float = 0.08,
        reframe_center_bias: float = 0.6,
        # -- misc --
        blur_strength: int = 20,
        blur_brightness: float = -0.15,
        output_quality: str = "medium",
        title_shadow_size: int = 2,
        title_letter_spacing: int = 0,
        title_fade_in_ms: int = 300,
        title_bar_opacity: float = 1.0,
    ) -> ClipGenerationResult:
        """Generate final clip with title and optional captions.

        The pipeline seeks directly into the source video and renders the
        composited output in a single FFmpeg invocation, avoiding an extra
        re-encoding pass from a separate extraction step.
        """
        intermediates: list[str] = []
        qp = QUALITY_PRESETS.get(output_quality, QUALITY_PRESETS["medium"])
        canvas_w, canvas_h = canvas_size_for_aspect_ratio(canvas_aspect_ratio)

        has_intro = (
            intro_file_path
            and intro_cfg
            and intro_cfg.get("enabled")
        )
        has_outro = (
            outro_file_path
            and outro_cfg
            and outro_cfg.get("enabled")
        )

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
        horizontal_padding = max(0, int(title_padding_x))
        max_text_w = max(120, base_title_width - (2 * horizontal_padding))
        title_lines = wrap_title(title, title_font_size, max_text_w)

        # Probe the source video once for resolution (used by layout computation).
        src_w, src_h = _probe_video_resolution(video_path)

        layout = compute_layout(
            video_path,
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
            source_width=src_w,
            source_height=src_h,
        )

        # Optional speaker reframe: crop the source to follow the face.
        effective_video_path = video_path
        compose_start_time = start_time
        compose_end_time = end_time
        requested_window_duration = max(0.1, float(end_time) - float(start_time))
        if reframe_enabled:
            try:
                from services.reframe.reframer import reframe_video as _reframe

                reframed_path = os.path.join(self.temp_dir, f"{clip_id}_reframed.mp4")
                logger.info(
                    "[%s] Reframe requested for window %.2f-%.2f (%.2fs)",
                    clip_id,
                    start_time,
                    end_time,
                    requested_window_duration,
                )
                reframe_ok = _reframe(
                    video_path,
                    reframed_path,
                    smoothing=reframe_smoothing,
                    center_bias=reframe_center_bias,
                    output_quality=output_quality,
                    start_time=start_time,
                    end_time=end_time,
                )
                if reframe_ok:
                    effective_video_path = reframed_path
                    compose_start_time = 0.0
                    compose_end_time = requested_window_duration
                    intermediates.append(reframed_path)
                    # Re-probe the reframed video for updated resolution.
                    src_w, src_h = _probe_video_resolution(reframed_path)
                    layout = compute_layout(
                        reframed_path,
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
                        source_width=src_w,
                        source_height=src_h,
                    )
                    logger.info(
                        "[%s] Reframe applied - using reframed source with compose seek %.2f-%.2f",
                        clip_id,
                        compose_start_time,
                        compose_end_time,
                    )
                else:
                    logger.info("[%s] Reframe skipped (no faces / unavailable)", clip_id)
            except Exception:
                logger.exception("[%s] Reframe failed — continuing with original source", clip_id)
                safe_remove(os.path.join(self.temp_dir, f"{clip_id}_reframed.mp4"))

        # If intro/outro concat runs afterward, keep this pass high-fidelity.
        overlay_qp = qp if not (has_intro or has_outro) else intermediate_quality_preset(qp)
        composited_path = os.path.join(self.temp_dir, f"{clip_id}_composited.mp4")
        compose_clip(
            effective_video_path,
            composited_path,
            style=background_style,
            canvas_w=layout["canvas_w"],
            canvas_h=layout["canvas_h"],
            vid_w=layout["vid_w"],
            vid_h=layout["vid_h"],
            vid_x=layout["vid_x"],
            vid_y=layout["vid_y"],
            blur_strength=blur_strength,
            video_scale_mode=video_scale_mode,
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
            qp=overlay_qp,
            overlay_file_path=overlay_file_path,
            overlay_cfg=overlay_cfg,
            background_color=background_color,
            background_image_path=background_image_path,
            start_time=compose_start_time,
            end_time=compose_end_time,
            blur_brightness=blur_brightness,
            title_shadow_size=title_shadow_size,
            title_letter_spacing=title_letter_spacing,
            title_fade_in_ms=title_fade_in_ms,
            title_bar_opacity=title_bar_opacity,
        )
        intermediates.append(composited_path)

        # Concatenate intro/outro if configured.
        if has_intro or has_outro:
            final_clip_path = os.path.join(self.temp_dir, f"{clip_id}_final.mp4")
            concat_intermediates = concat_intro_outro(
                composited_path,
                final_clip_path,
                qp,
                intro_file_path=intro_file_path,
                intro_cfg=intro_cfg,
                outro_file_path=outro_file_path,
                outro_cfg=outro_cfg,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            intermediates.extend(concat_intermediates)
        else:
            final_clip_path = composited_path
            # Remove composited_path from intermediates since it IS the final.
            intermediates = [p for p in intermediates if p != composited_path]

        file_size = os.path.getsize(final_clip_path)
        return {
            "clip_path": final_clip_path,
            "file_size": file_size,
            "intermediates": intermediates,
        }


__all__ = ["ClipGenerator", "compute_video_position"]

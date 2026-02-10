import logging
import os
from typing import Any

import ffmpeg
from config import TEMP_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canvas constants (9:16 portrait for short-form platforms)
# ---------------------------------------------------------------------------
CANVAS_W = 1080
CANVAS_H = 1920

TITLE_BAR_V_PAD = 16  # vertical padding inside the title bar
TITLE_GAP = 12  # gap between the title bar and the video

# x264 quality presets keyed by the API ``outputQuality`` value
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "low": {"crf": 28, "preset": "veryfast"},
    "medium": {"crf": 23, "preset": "medium"},
    "high": {"crf": 18, "preset": "slow"},
}

# Average character width as a fraction of font size (proportional fonts).
# Used to estimate line length for word-wrapping.
_CHAR_WIDTH_RATIO = 0.52


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_title(title: str, font_size: int, max_width: int) -> list[str]:
    """Word-wrap *title* so it fits within *max_width* pixels.

    Returns a **list of lines**.  Each line is rendered by a separate
    ``drawtext`` filter so we never embed newline characters (which
    ffmpeg renders as □ boxes).
    """
    avg_char_w = font_size * _CHAR_WIDTH_RATIO
    max_chars = max(10, int(max_width / avg_char_w))

    words = title.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        candidate = f"{current_line} {word}".strip() if current_line else word
        if len(candidate) <= max_chars:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Limit to 3 lines max; truncate the last line if needed
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1][: max_chars - 3].rstrip() + "..."

    return lines


def compute_video_position(
    src_w: int,
    src_h: int,
    width_pct: int = 100,
    position_y: str = "middle",
) -> tuple[int, int, int, int]:
    """Compute video dimensions and position on the 9:16 canvas.

    Parameters match the API ``VideoLayout`` model.
    Returns ``(vid_w, vid_h, vid_x, vid_y)``.
    """
    vid_w = int(CANVAS_W * max(10, min(100, width_pct)) / 100)
    vid_w -= vid_w % 2
    vid_h = int(src_h * vid_w / src_w)
    vid_h -= vid_h % 2
    vid_x = (CANVAS_W - vid_w) // 2

    if position_y == "middle":
        vid_y = (CANVAS_H - vid_h) // 2
    elif position_y == "top":
        vid_y = 0
    elif position_y == "bottom":
        vid_y = CANVAS_H - vid_h
    else:
        vid_y = int(position_y)

    vid_y = max(0, min(vid_y, CANVAS_H - vid_h))
    return vid_w, vid_h, vid_x, vid_y


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
        # -- video layout --
        video_width_pct: int = 100,
        video_position_y: str = "middle",
        # -- title style --
        title_show: bool = True,
        title_font_size: int = 48,
        title_font_color: str = "white",
        title_font_family: str = "",
        title_align: str = "left",
        title_stroke_width: int = 0,
        title_stroke_color: str = "black",
        title_bar_enabled: bool = True,
        title_bar_color: str = "black@0.5",
        title_padding_x: int = 16,
        title_position_y: str = "above_video",
        # -- captions --
        caption_ass_path: str | None = None,
        # -- misc --
        blur_strength: int = 20,
        output_quality: str = "medium",
    ) -> dict:
        """Generate final portrait clip with overlay.

        Returns dict with ``clip_path``, ``thumbnail_path``, ``file_size``
        and an ``intermediates`` list of temp files.
        """
        intermediates: list[str] = []
        qp = QUALITY_PRESETS.get(output_quality, QUALITY_PRESETS["medium"])

        # -- Word-wrap the title ------------------------------------------
        max_text_w = CANVAS_W - 6 * title_padding_x
        title_lines = _wrap_title(title, title_font_size, max_text_w)

        # 1. Extract clip segment
        raw_clip_path = os.path.join(self.temp_dir, f"{clip_id}_raw.mp4")
        self._extract_segment(video_path, start_time, end_time, raw_clip_path, qp)
        intermediates.append(raw_clip_path)

        # 2. Compute layout (pass line_count so the title bar height accounts for wrapping)
        layout = self._compute_layout(
            raw_clip_path,
            video_width_pct,
            video_position_y,
            title_font_size,
            title_padding_x,
            title_position_y,
            len(title_lines),
        )

        # 3. Create portrait background with scaled & positioned video
        bg_clip_path = os.path.join(self.temp_dir, f"{clip_id}_bg.mp4")
        self._create_portrait_background(
            raw_clip_path,
            bg_clip_path,
            background_style,
            layout["vid_w"],
            layout["vid_h"],
            layout["vid_x"],
            layout["vid_y"],
            blur_strength,
            qp,
        )
        intermediates.append(bg_clip_path)

        # 4. Add all overlays (title + captions) in one encode pass
        final_clip_path = os.path.join(self.temp_dir, f"{clip_id}_final.mp4")
        self._add_overlays(
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
            title_padding_x=layout["title_padding_x"],
            title_bar_y=layout["title_bar_y"],
            title_text_y=layout["title_text_y"],
            title_bar_h=layout["title_bar_h"],
            caption_ass_path=caption_ass_path,
            qp=qp,
        )

        # 5. Generate thumbnail
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
    # Layout computation
    # ------------------------------------------------------------------
    def _compute_layout(
        self,
        video_path: str,
        video_width_pct: int,
        video_position_y: str,
        title_font_size: int,
        title_padding_x: int,
        title_position_y: str,
        title_line_count: int = 1,
    ) -> dict:
        """Probe the source clip and return pixel values for positioning."""

        probe = ffmpeg.probe(video_path)
        v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        src_w = int(v_stream["width"])
        src_h = int(v_stream["height"])

        vid_w, vid_h, vid_x, vid_y = compute_video_position(
            src_w, src_h, video_width_pct, video_position_y
        )

        # -- Title bar dimensions (accounts for wrapped lines) -------------
        single_line_h = title_font_size + 2 * TITLE_BAR_V_PAD
        title_bar_h = single_line_h + (title_line_count - 1) * int(
            title_font_size * 1.3
        )

        if title_position_y == "above_video":
            title_bar_y = max(0, vid_y - title_bar_h - TITLE_GAP)
        elif title_position_y == "top":
            title_bar_y = 0
        elif title_position_y == "bottom":
            title_bar_y = CANVAS_H - title_bar_h
        else:
            title_bar_y = max(0, int(title_position_y))

        title_text_y = title_bar_y + TITLE_BAR_V_PAD

        logger.info(
            "Layout: video %dx%d at (%d,%d)  title bar y=%d h=%d  pad_x=%d  lines=%d",
            vid_w,
            vid_h,
            vid_x,
            vid_y,
            title_bar_y,
            title_bar_h,
            title_padding_x,
            title_line_count,
        )

        return {
            "vid_w": vid_w,
            "vid_h": vid_h,
            "vid_x": vid_x,
            "vid_y": vid_y,
            "title_padding_x": title_padding_x,
            "title_bar_y": title_bar_y,
            "title_bar_h": title_bar_h,
            "title_text_y": title_text_y,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_segment(
        self,
        input_path: str,
        start: float,
        end: float,
        output_path: str,
        qp: dict,
    ):
        """Extract video segment."""
        duration = end - start
        logger.info("Extracting segment %.2f–%.2f (%.2fs)", start, end, duration)

        (
            ffmpeg.input(input_path, ss=start, t=duration)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
            )
            .overwrite_output()
            .run(quiet=True)
        )

    def _create_portrait_background(
        self,
        input_path: str,
        output_path: str,
        style: str,
        vid_w: int,
        vid_h: int,
        vid_x: int,
        vid_y: int,
        blur_strength: int,
        qp: dict,
    ):
        """Create 9:16 portrait canvas with the video scaled & positioned."""
        logger.info(
            "Creating portrait background (style=%s, blur=%d)", style, blur_strength
        )

        if style == "blur":
            bg_temp = output_path + ".bg.mp4"

            # Blurred full-canvas background
            (
                ffmpeg.input(input_path)
                .filter(
                    "scale", CANVAS_W, CANVAS_H, force_original_aspect_ratio="increase"
                )
                .filter("crop", CANVAS_W, CANVAS_H)
                .filter("boxblur", blur_strength)
                .output(bg_temp)
                .overwrite_output()
                .run(quiet=True)
            )

            # Scale source video & overlay at computed position
            bg = ffmpeg.input(bg_temp)
            video = ffmpeg.input(input_path).filter("scale", vid_w, vid_h)

            (
                ffmpeg.filter([bg, video], "overlay", vid_x, vid_y)
                .output(
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    crf=qp["crf"],
                    preset=qp["preset"],
                )
                .overwrite_output()
                .run(quiet=True)
            )

            self._safe_remove(bg_temp)

        else:  # solid_color / gradient
            (
                ffmpeg.input(input_path)
                .filter("scale", vid_w, vid_h)
                .filter("pad", CANVAS_W, CANVAS_H, vid_x, vid_y, "black")
                .output(
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    crf=qp["crf"],
                    preset=qp["preset"],
                )
                .overwrite_output()
                .run(quiet=True)
            )

    def _add_overlays(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        *,
        title_lines: list[str],
        title_show: bool,
        title_font_size: int,
        title_font_color: str,
        title_font_family: str,
        title_align: str,
        title_stroke_width: int,
        title_stroke_color: str,
        title_bar_enabled: bool,
        title_bar_color: str,
        title_padding_x: int,
        title_bar_y: int,
        title_text_y: int,
        title_bar_h: int,
        caption_ass_path: str | None,
        qp: dict,
    ):
        """Draw title bar + text + captions in a single encode pass."""
        logger.info(
            "Adding overlays (title_show=%s, bar=%s, color=%s, lines=%d, captions=%s)",
            title_show,
            title_bar_enabled,
            title_font_color,
            len(title_lines),
            bool(caption_ass_path),
        )

        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)

        stream = video

        # -- Title ---------------------------------------------------------
        if title_show and title_lines:
            # Full-width background bar
            if title_bar_enabled:
                stream = stream.drawbox(
                    x=0,
                    y=title_bar_y,
                    width="iw",
                    height=title_bar_h,
                    color=title_bar_color,
                    t="fill",
                )

            # Line height for multi-line spacing (font size * 1.3)
            line_height = int(title_font_size * 1.3)

            # One drawtext filter per line to avoid □ glyph from \n
            for i, line_text in enumerate(title_lines):
                # Text x-position (recalculated per line for center align)
                if title_align == "center":
                    text_x = "(w-text_w)/2"
                else:
                    text_x = str(title_padding_x)

                line_y = title_text_y + i * line_height

                dt_kwargs: dict[str, Any] = {
                    "text": line_text,
                    "fontsize": title_font_size,
                    "fontcolor": title_font_color,
                    "x": text_x,
                    "y": str(line_y),
                }
                if title_font_family:
                    dt_kwargs["font"] = title_font_family

                # Stroke (outline) or shadow -- stroke takes precedence
                if title_stroke_width > 0:
                    dt_kwargs["borderw"] = title_stroke_width
                    dt_kwargs["bordercolor"] = title_stroke_color
                else:
                    dt_kwargs["shadowcolor"] = "black@0.7"
                    dt_kwargs["shadowx"] = 2
                    dt_kwargs["shadowy"] = 2

                stream = stream.drawtext(**dt_kwargs)

        # -- Captions (ASS subtitles) --------------------------------------
        if caption_ass_path and os.path.isfile(caption_ass_path):
            # Escape the path for ffmpeg filter strings:
            # - Windows paths use backslashes → convert to forward slashes
            # - Colons (C:) are ffmpeg filter syntax → escape as \:
            escaped_path = caption_ass_path.replace("\\", "/").replace(":", "\\:")
            stream = stream.filter("subtitles", escaped_path)

        # -- Encode --------------------------------------------------------
        try:
            (
                ffmpeg.output(
                    stream,
                    audio.audio,
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    crf=qp["crf"],
                    preset=qp["preset"],
                    audio_bitrate="192k",
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(capture_stderr=True)
            )
        except ffmpeg.Error as e:
            stderr_output = (
                e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            )
            logger.error("FFmpeg overlay failed:\n%s", stderr_output)
            raise

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

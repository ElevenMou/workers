"""Layout helpers for clip generation."""

import logging

import ffmpeg

from services.clips.constants import (
    DEFAULT_CANVAS_ASPECT_RATIO,
    CHAR_WIDTH_RATIO,
    TITLE_BAR_V_PAD,
    TITLE_GAP,
    canvas_size_for_aspect_ratio,
    normalize_video_scale_mode,
)
from services.clips.models import ClipLayout

logger = logging.getLogger(__name__)


def wrap_title(title: str, font_size: int, max_width: int) -> list[str]:
    """Word-wrap title text to fit within a given pixel width."""
    avg_char_w = font_size * CHAR_WIDTH_RATIO
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

    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1][: max_chars - 3].rstrip() + "..."

    return lines


def compute_video_position(
    src_w: int,
    src_h: int,
    width_pct: int = 100,
    position_y: str = "middle",
    *,
    canvas_w: int | None = None,
    canvas_h: int | None = None,
    video_scale_mode: str = "fit",
) -> tuple[int, int, int, int]:
    """Compute scaled video size and x/y position on the target canvas."""
    if canvas_w is None or canvas_h is None:
        canvas_w, canvas_h = canvas_size_for_aspect_ratio(DEFAULT_CANVAS_ASPECT_RATIO)

    clamped_width_pct = max(10, min(100, int(width_pct)))
    vid_w = int(canvas_w * clamped_width_pct / 100)
    vid_w = max(2, vid_w - (vid_w % 2))

    scale_mode = normalize_video_scale_mode(video_scale_mode)
    if scale_mode == "fill":
        # Fill mode targets a box that follows the canvas ratio.
        vid_h = int(vid_w * canvas_h / canvas_w)
    else:
        # Fit mode preserves the source aspect in the box width.
        vid_h = int(src_h * vid_w / src_w)
    vid_h = max(2, vid_h - (vid_h % 2))
    vid_h = min(vid_h, canvas_h)

    vid_x = (canvas_w - vid_w) // 2

    if position_y == "middle":
        vid_y = (canvas_h - vid_h) // 2
    elif position_y == "top":
        vid_y = 0
    elif position_y == "bottom":
        vid_y = canvas_h - vid_h
    else:
        vid_y = int(position_y)

    vid_y = max(0, min(vid_y, canvas_h - vid_h))
    return vid_w, vid_h, vid_x, vid_y


def compute_layout(
    video_path: str,
    video_width_pct: int,
    video_position_y: str,
    title_font_size: int,
    title_padding_x: int,
    title_position_y: str,
    canvas_aspect_ratio: str,
    video_scale_mode: str,
    title_line_count: int = 1,
) -> ClipLayout:
    """Probe the source clip and return concrete layout pixel values."""
    canvas_w, canvas_h = canvas_size_for_aspect_ratio(canvas_aspect_ratio)
    probe = ffmpeg.probe(video_path)
    v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    src_w = int(v_stream["width"])
    src_h = int(v_stream["height"])

    vid_w, vid_h, vid_x, vid_y = compute_video_position(
        src_w,
        src_h,
        video_width_pct,
        video_position_y,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        video_scale_mode=video_scale_mode,
    )

    single_line_h = title_font_size + 2 * TITLE_BAR_V_PAD
    title_bar_h = single_line_h + (title_line_count - 1) * int(title_font_size * 1.3)

    if title_position_y == "above_video":
        title_bar_y = max(0, vid_y - title_bar_h - TITLE_GAP)
    elif title_position_y == "top":
        title_bar_y = 0
    elif title_position_y == "bottom":
        title_bar_y = canvas_h - title_bar_h
    else:
        title_bar_y = max(0, int(title_position_y))

    title_text_y = title_bar_y + TITLE_BAR_V_PAD

    logger.info(
        "Layout (%s/%s): video %dx%d at (%d,%d)  title bar y=%d h=%d  pad_x=%d  lines=%d",
        canvas_aspect_ratio,
        video_scale_mode,
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
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "vid_w": vid_w,
        "vid_h": vid_h,
        "vid_x": vid_x,
        "vid_y": vid_y,
        "title_padding_x": title_padding_x,
        "title_bar_y": title_bar_y,
        "title_bar_h": title_bar_h,
        "title_text_y": title_text_y,
    }

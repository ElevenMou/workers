"""ASS rendering orchestrator for caption styles."""

import logging
from typing import Any

from services.captions.line_builders import (
    build_animated_lines,
    build_grouped_lines,
    build_progressive_lines,
    build_punctuated_lines,
    build_static_lines,
    build_two_line_grouped_lines,
    build_word_lines,
)
from services.captions.presets import (
    _normalize_animation,
    _normalize_font_case,
    normalize_caption_style,
)

logger = logging.getLogger(__name__)


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert ``#RRGGBB`` or ``#AARRGGBB`` to ASS ``&HAABBGGRR`` format."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}&".upper()
    if len(h) == 8:
        a, r, g, b = h[0:2], h[2:4], h[4:6], h[6:8]
        return f"&H{a}{b}{g}{r}&".upper()
    return "&H00FFFFFF&"


def _color_name_to_ass(name: str) -> str:
    """Best-effort named color to ASS hex, with white fallback."""
    name_map = {
        "white": "&H00FFFFFF&",
        "black": "&H00000000&",
        "red": "&H000000FF&",
        "green": "&H0000FF00&",
        "blue": "&H00FF0000&",
        "yellow": "&H0000FFFF&",
        "cyan": "&H00FFFF00&",
        "magenta": "&H00FF00FF&",
    }
    if name.startswith("#"):
        return _hex_to_ass_color(name)
    return name_map.get(name.lower(), "&H00FFFFFF&")


def _safe_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


_CAPTION_INNER_PAD = 30
_CAPTION_OUTER_GAP = 20


def _caption_placement(
    position: str,
    font_size: int,
    canvas_h: int,
    vid_y: int | None,
    vid_h: int | None,
) -> tuple[int, int]:
    """Return ``(ass_alignment, margin_v)`` for the given caption position."""
    if vid_y is not None and vid_h is not None:
        vid_bottom = vid_y + vid_h

        if position in {"auto", "bottom"}:
            alignment = 2
            margin_v = canvas_h - vid_bottom + _CAPTION_INNER_PAD
        elif position == "middle":
            alignment = 8
            margin_v = vid_y + (vid_h - font_size) // 2
        elif position == "below_video":
            alignment = 8
            margin_v = vid_bottom + _CAPTION_OUTER_GAP
        elif position == "above_video":
            alignment = 2
            margin_v = canvas_h - vid_y + _CAPTION_OUTER_GAP
        elif position == "top":
            alignment = 8
            margin_v = vid_y + _CAPTION_INNER_PAD
        else:
            alignment = 2
            margin_v = 80
    else:
        effective = "bottom" if position == "auto" else position
        alignment = {"bottom": 2, "middle": 5, "top": 8}.get(effective, 2)
        margin_v = 80 if effective == "bottom" else 40

    return alignment, max(0, margin_v)


def render_ass(
    segments: list[dict],
    *,
    style: str = "animated",
    animation: str = "none",
    font_size: int = 42,
    font_color: str = "white",
    font_family: str = "",
    font_weight: str = "bold",
    font_case: str = "as_typed",
    italic: bool = False,
    underline: bool = False,
    stroke_color: str = "black",
    stroke_thickness: int = 3,
    shadow_color: str = "#000000AA",
    shadow_x: int = 2,
    shadow_y: int = 2,
    shadow_blur: int = 2,
    highlight_color: str = "#FFD700",
    position: str = "bottom",
    lines_per_page: int = 1,
    max_words_per_line: int = 5,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    vid_y: int | None = None,
    vid_h: int | None = None,
    output_path: str,
) -> str:
    """Generate an ASS subtitle file and return its path."""
    primary_color = _color_name_to_ass(font_color)
    secondary_color = _color_name_to_ass(highlight_color)
    outline_color = _color_name_to_ass(stroke_color)
    back_color = _color_name_to_ass(shadow_color)
    fname = font_family or "Arial"
    normalized_case = _normalize_font_case(font_case)
    normalized_animation = _normalize_animation(animation)

    bold_flag = (
        -1
        if str(font_weight).strip().lower()
        in {"bold", "semibold", "black", "heavy", "700", "800", "900"}
        else 0
    )
    italic_flag = -1 if italic else 0
    underline_flag = -1 if underline else 0
    safe_outline = _safe_int(stroke_thickness, default=3, minimum=0)
    safe_shadow = max(
        _safe_int(shadow_x, default=2, minimum=0),
        _safe_int(shadow_y, default=2, minimum=0),
        _safe_int(shadow_blur, default=2, minimum=0),
    )

    alignment, margin_v = _caption_placement(position, font_size, canvas_h, vid_y, vid_h)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {canvas_w}\n"
        f"PlayResY: {canvas_h}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fname},{font_size},{primary_color},{secondary_color},"
        f"{outline_color},{back_color},{bold_flag},{italic_flag},{underline_flag},0,100,100,0,0,1,{safe_outline},{safe_shadow},"
        f"{alignment},20,20,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    normalized_style = normalize_caption_style(style)
    if style and style != normalized_style:
        logger.info(
            "Caption style normalized from '%s' to '%s'",
            style,
            normalized_style,
        )

    safe_max_words_per_line = _safe_int(max_words_per_line, default=5, minimum=1)
    safe_lines_per_page = _safe_int(lines_per_page, default=1, minimum=1)

    lines: list[str] = []
    if normalized_style == "animated":
        lines = build_animated_lines(
            segments,
            safe_max_words_per_line,
            font_case=normalized_case,
            animation=normalized_animation,
        )
    elif normalized_style == "grouped":
        lines = build_grouped_lines(
            segments,
            safe_max_words_per_line,
            transform=normalized_case,
            lines_per_page=safe_lines_per_page,
            animation=normalized_animation,
        )
    elif normalized_style == "word_by_word":
        lines = build_word_lines(
            segments,
            font_case=normalized_case,
            animation=normalized_animation,
        )
    elif normalized_style == "progressive":
        lines = build_progressive_lines(
            segments,
            safe_max_words_per_line,
            font_case=normalized_case,
            animation=normalized_animation,
        )
    elif normalized_style == "two_line":
        lines = build_two_line_grouped_lines(
            segments,
            safe_max_words_per_line,
            transform=normalized_case,
            animation=normalized_animation,
        )
    elif normalized_style == "punctuated":
        lines = build_punctuated_lines(
            segments,
            font_case=normalized_case,
            animation=normalized_animation,
        )
    elif normalized_style in {"uppercase", "lowercase"}:
        lines = build_static_lines(
            segments,
            transform=normalized_style,
            lines_per_page=safe_lines_per_page,
            animation=normalized_animation,
        )
    elif normalized_style == "headline":
        lines = build_grouped_lines(
            segments,
            safe_max_words_per_line,
            transform="headline",
            lines_per_page=safe_lines_per_page,
            animation=normalized_animation,
        )
    else:
        lines = build_static_lines(
            segments,
            transform=normalized_case,
            lines_per_page=safe_lines_per_page,
            animation=normalized_animation,
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in lines:
            f.write(line + "\n")

    logger.info("Wrote %d caption lines to %s", len(lines), output_path)
    return output_path

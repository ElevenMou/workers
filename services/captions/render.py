"""Backward-compatible ASS rendering entrypoint."""

from __future__ import annotations

from typing import Any

from services.captions.ass_generator import generate_ass_file
from services.captions.caption_presets import normalize_caption_style, to_ass_color
from services.captions.positioning import compute_video_anchored_margin_v


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def render_ass(
    segments: list[dict[str, Any]],
    *,
    style: str = "clean_minimal",
    animation: str = "none",
    font_size: int = 68,
    font_color: str = "&H00FFFFFF",
    font_family: str = "Montserrat-Bold",
    font_weight: str = "bold",
    font_case: str = "as_typed",
    italic: bool = False,
    underline: bool = False,
    stroke_color: str = "&H00000000",
    stroke_thickness: int = 3,
    shadow_color: str = "&H80000000",
    shadow_x: int = 1,
    shadow_y: int = 1,
    shadow_blur: int = 1,
    highlight_color: str = "&H0000C8FF",
    position: str = "auto",
    lines_per_page: int = 2,
    max_words_per_line: int = 5,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    vid_y: int | None = None,
    vid_h: int | None = None,
    output_path: str,
) -> str:
    """Render ASS from normalized segment list with compatibility arguments.

    The legacy style/mode args are mapped to the new preset architecture.
    """
    del font_weight, shadow_x, shadow_y

    preset_name = normalize_caption_style(style)
    words_to_char_ratio = 6
    max_chars = max(12, int(max_words_per_line) * words_to_char_ratio)
    if lines_per_page > 1:
        max_chars = int(max_chars * 1.1)

    overrides: dict[str, Any] = {
        "animation": animation,
        "font_name": font_family or "Montserrat-Bold",
        "font_size": int(font_size),
        "primary_color": to_ass_color(font_color),
        "secondary_color": to_ass_color(highlight_color),
        "outline_color": to_ass_color(stroke_color, fallback="&H00000000"),
        "back_color": to_ass_color(shadow_color, fallback="&H80000000"),
        "outline": max(0, int(stroke_thickness)),
        "shadow": max(0, int(shadow_blur)),
        "position": position,
        "max_chars_per_line": max_chars,
        "max_lines": max(1, int(lines_per_page)),
        "uppercase": str(font_case).strip().lower() == "uppercase",
        "italic": bool(italic),
        "underline": bool(underline),
    }

    anchored_margin = compute_video_anchored_margin_v(
        position=position,
        canvas_h=canvas_h,
        vid_y=vid_y,
        vid_h=vid_h,
        inset=_to_int(overrides.get("margin_v"), 80),
    )
    if anchored_margin is not None:
        overrides["margin_v"] = anchored_margin

    video_aspect_ratio = "16:9" if canvas_w >= canvas_h else "9:16"
    transcript_json = {"segments": segments}
    return generate_ass_file(
        transcript_json=transcript_json,
        preset_name=preset_name,
        output_path=output_path,
        video_aspect_ratio=video_aspect_ratio,
        overrides=overrides,
    )


__all__ = ["render_ass"]

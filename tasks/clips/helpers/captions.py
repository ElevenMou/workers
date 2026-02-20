"""Shared caption rendering helper for task modules."""

from __future__ import annotations

import logging
import os
from typing import Any

from services.caption_renderer import (
    extract_clip_segments,
    generate_ass_file,
    normalize_caption_style,
    resolve_preset,
    to_ass_color,
)
from services.captions.positioning import compute_video_anchored_margin_v


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _overrides_from_layout(cap_cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate layout caption fields into preset override keys."""
    overrides: dict[str, Any] = {}

    font_size = cap_cfg.get("fontSize")
    if isinstance(font_size, (int, float)):
        overrides["font_size"] = int(font_size)

    font_family = cap_cfg.get("fontFamily")
    if isinstance(font_family, str) and font_family.strip():
        overrides["font_name"] = font_family.strip()

    font_color = cap_cfg.get("fontColor")
    if isinstance(font_color, str) and font_color.strip():
        overrides["primary_color"] = to_ass_color(font_color)

    highlight_color = cap_cfg.get("highlightColor")
    if isinstance(highlight_color, str) and highlight_color.strip():
        overrides["secondary_color"] = to_ass_color(highlight_color)

    stroke_color = cap_cfg.get("strokeColor")
    if isinstance(stroke_color, str) and stroke_color.strip():
        overrides["outline_color"] = to_ass_color(stroke_color, fallback="&H00000000")

    shadow_color = cap_cfg.get("shadowColor")
    if isinstance(shadow_color, str) and shadow_color.strip():
        overrides["back_color"] = to_ass_color(shadow_color, fallback="&H80000000")

    stroke_thickness = cap_cfg.get("strokeThickness")
    if isinstance(stroke_thickness, (int, float)):
        overrides["outline"] = max(0, int(stroke_thickness))

    shadow_blur = cap_cfg.get("shadowBlur")
    if isinstance(shadow_blur, (int, float)):
        overrides["shadow"] = max(0, int(shadow_blur))

    position = cap_cfg.get("position")
    if isinstance(position, str) and position.strip():
        overrides["position"] = position.strip().lower()

    max_chars = cap_cfg.get("maxCharsPerCaption")
    if isinstance(max_chars, (int, float)):
        overrides["max_chars_per_line"] = max(8, int(max_chars))

    max_lines = cap_cfg.get("maxLines")
    if isinstance(max_lines, (int, float)):
        overrides["max_lines"] = max(1, int(max_lines))

    line_delay = cap_cfg.get("lineDelay")
    if isinstance(line_delay, (int, float)):
        overrides["line_delay"] = max(0.0, float(line_delay))

    if "wordHighlight" in cap_cfg:
        overrides["word_highlight"] = bool(cap_cfg.get("wordHighlight"))
    if "backgroundBox" in cap_cfg:
        overrides["background_box"] = bool(cap_cfg.get("backgroundBox"))
    if "uppercase" in cap_cfg:
        overrides["uppercase"] = bool(cap_cfg.get("uppercase"))
    elif str(cap_cfg.get("fontCase", "")).lower() == "uppercase":
        overrides["uppercase"] = True

    animation = cap_cfg.get("animation")
    if isinstance(animation, str) and animation.strip():
        overrides["animation"] = animation.strip()

    # Map the new style field ("grouped", "word_by_word", "karaoke").
    # When style is "karaoke", also enable word_highlight so the ASS
    # generator uses \kf tags for progressive fill.
    style = cap_cfg.get("style")
    if isinstance(style, str) and style.strip():
        normalized_style = style.strip().lower()
        if normalized_style in {"grouped", "word_by_word", "karaoke"}:
            overrides["style"] = normalized_style
            if normalized_style == "karaoke":
                overrides["word_highlight"] = True

    return overrides


def build_caption_ass(
    *,
    job_id: str,
    clip_id: str,
    transcript: dict[str, Any] | None,
    cap_cfg: dict[str, Any],
    start_time: float,
    end_time: float,
    canvas_w: int,
    canvas_h: int,
    vid_y: int,
    vid_h: int,
    video_aspect_ratio: str,
    work_dir: str,
    logger: logging.Logger,
) -> str | None:
    """Render caption ASS file from transcript and caption config."""
    if not cap_cfg.get("show"):
        logger.info(
            "[%s] Captions disabled by layout config (show=%r, preset=%r)",
            job_id,
            cap_cfg.get("show"),
            cap_cfg.get("presetName") or cap_cfg.get("style"),
        )
        return None
    if not transcript or not transcript.get("segments"):
        logger.warning("[%s] Captions requested but no transcript available", job_id)
        return None

    requested_preset = (
        cap_cfg.get("presetName")
        or cap_cfg.get("preset")
        or cap_cfg.get("style")
        or "clean"
    )
    preset_name = normalize_caption_style(str(requested_preset))
    logger.info("[%s] Building caption preset '%s' ...", job_id, preset_name)

    overrides = _overrides_from_layout(cap_cfg)
    resolved_preset = resolve_preset(preset_name, overrides=overrides)
    requested_position = str(
        overrides.get("position") or cap_cfg.get("position") or "auto"
    ).strip().lower()
    if requested_position in {"top", "bottom"}:
        inset = _to_int(
            resolved_preset.get(
                "safe_margin_y" if requested_position == "top" else "margin_v"
            ),
            _to_int(resolved_preset.get("margin_v"), 80),
        )
        anchored_margin = compute_video_anchored_margin_v(
            position=requested_position,
            canvas_h=canvas_h,
            vid_y=vid_y,
            vid_h=vid_h,
            inset=inset,
        )
        if anchored_margin is not None:
            overrides["margin_v"] = anchored_margin
            logger.info(
                "[%s] Caption position '%s' anchored to video bounds (margin_v=%d)",
                job_id,
                requested_position,
                anchored_margin,
            )

    segments = extract_clip_segments(transcript, start_time, end_time)
    if not segments:
        return None

    return generate_ass_file(
        transcript_json={"segments": segments},
        preset_name=preset_name,
        output_path=os.path.join(work_dir, f"{clip_id}.ass"),
        video_aspect_ratio=video_aspect_ratio,
        overrides=overrides,
    )

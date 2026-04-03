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
from tasks.videos.transcript import transcript_is_rtl

_STYLE_MODES = {"grouped", "karaoke", "highlight", "highlight_box"}


def resolve_caption_style_mode(cap_cfg: dict[str, Any]) -> str:
    """Determine the effective caption style mode from layout config."""
    requested_preset = (
        cap_cfg.get("presetName")
        or cap_cfg.get("preset")
        or cap_cfg.get("style")
        or "clean"
    )
    preset_name = normalize_caption_style(str(requested_preset))
    preset_defaults = resolve_preset(preset_name)
    return _normalize_style_mode(cap_cfg.get("style"), preset_defaults.get("style"))


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pick_numeric(cap_cfg: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        candidate = _as_number(cap_cfg.get(key))
        if candidate is not None:
            return candidate
    return None


def _normalize_style_mode(style_value: Any, fallback: Any) -> str:
    requested = str(style_value or "").strip().lower()
    if requested in _STYLE_MODES:
        return requested

    fallback_mode = str(fallback or "").strip().lower()
    if fallback_mode in _STYLE_MODES:
        return fallback_mode
    return "grouped"


def resolve_rtl_safe_caption_style(
    style_mode: str,
    transcript: dict[str, Any] | None,
) -> str:
    normalized = _normalize_style_mode(style_mode, "grouped")
    if transcript_is_rtl(transcript):
        return "grouped"
    return normalized


def _normalize_font_weight_token(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _weight_implies_bold(weight_token: str) -> bool | None:
    if not weight_token:
        return None
    if weight_token in {"700", "800", "900", "bold", "black"}:
        return True
    if weight_token in {"100", "200", "300", "400", "500", "normal", "regular", "light"}:
        return False
    return None


def _resolve_ass_font_name(
    *,
    font_family: str,
    font_weight: Any,
    bold_hint: bool,
) -> str:
    raw_family = font_family.strip()
    if not raw_family:
        return "Montserrat-Bold"

    # Keep explicit variant names unchanged (for example "Montserrat-Bold").
    if "-" in raw_family:
        return raw_family

    normalized_family = raw_family.lower()
    weight_token = _normalize_font_weight_token(font_weight)
    weight_is_bold = _weight_implies_bold(weight_token)
    is_bold = bold_hint if weight_is_bold is None else weight_is_bold

    if normalized_family == "montserrat":
        if weight_token in {"300", "light"}:
            return "Montserrat-Light"
        if weight_token in {"900", "black"}:
            return "Montserrat-Black"
        return "Montserrat-Bold" if is_bold else "Montserrat"

    if normalized_family == "poppins":
        if weight_token in {"900", "black"}:
            return "Poppins-Black"
        return "Poppins-Bold" if is_bold else "Poppins"

    if normalized_family == "inter":
        return "Inter-Bold" if is_bold else "Inter"

    if normalized_family == "roboto":
        return "Roboto-Bold" if is_bold else "Roboto"

    if normalized_family == "oswald":
        return "Oswald-Bold" if is_bold else "Oswald"

    if normalized_family == "space mono":
        return "SpaceMono-Bold" if is_bold else "SpaceMono-Regular"

    return raw_family


def _normalize_case_mode(cap_cfg: dict[str, Any], *, preset_defaults: dict[str, Any]) -> str:
    case_mode = str(cap_cfg.get("fontCase") or "").strip().lower()
    if case_mode in {"uppercase", "lowercase", "as_typed"}:
        return case_mode
    if "uppercase" in cap_cfg:
        return "uppercase" if bool(cap_cfg.get("uppercase")) else "as_typed"
    preset_case_mode = str(preset_defaults.get("font_case") or "").strip().lower()
    if preset_case_mode in {"uppercase", "lowercase", "as_typed"}:
        return preset_case_mode
    return "uppercase" if bool(preset_defaults.get("uppercase", False)) else "as_typed"


def _normalize_word_highlight(
    cap_cfg: dict[str, Any],
    *,
    style_mode: str,
    preset_defaults: dict[str, Any],
) -> bool:
    if "wordHighlight" in cap_cfg:
        return bool(cap_cfg.get("wordHighlight"))
    if style_mode == "karaoke":
        return True
    return bool(preset_defaults.get("word_highlight", False))


def _default_rtl_caption_font_name() -> str:
    configured = str(os.getenv("RTL_CAPTION_FONT_FAMILY", "")).strip()
    if configured:
        return configured
    return "Segoe UI" if os.name == "nt" else "Noto Sans Arabic"


def _overrides_from_layout(
    cap_cfg: dict[str, Any],
    *,
    canvas_w: int,
    canvas_h: int,
    preset_name: str = "clean",
    transcript: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate layout caption fields into normalized preset override keys."""
    overrides: dict[str, Any] = {}
    preset_defaults = resolve_preset(preset_name)
    rtl_transcript = transcript_is_rtl(transcript)

    style_mode = resolve_rtl_safe_caption_style(
        _normalize_style_mode(cap_cfg.get("style"), preset_defaults.get("style")),
        transcript,
    )
    overrides["style"] = style_mode
    overrides["word_highlight"] = _normalize_word_highlight(
        cap_cfg,
        style_mode=style_mode,
        preset_defaults=preset_defaults,
    )
    case_mode = _normalize_case_mode(
        cap_cfg,
        preset_defaults=preset_defaults,
    )
    overrides["font_case"] = case_mode
    overrides["uppercase"] = case_mode == "uppercase"
    if rtl_transcript:
        overrides["font_case"] = "as_typed"
        overrides["uppercase"] = False
        overrides["word_highlight"] = False

    max_chars = _pick_numeric(cap_cfg, "maxCharsPerCaption", "maxCharsPerLine")
    if max_chars is None:
        max_words_per_line = _pick_numeric(cap_cfg, "maxWordsPerLine")
        if max_words_per_line is not None:
            max_chars = max_words_per_line * 6.0
    if max_chars is None:
        max_chars = float(_to_int(preset_defaults.get("max_chars_per_line"), 30))
    overrides["max_chars_per_line"] = max(8, int(max_chars))

    max_lines = _pick_numeric(cap_cfg, "maxLines", "linesPerPage")
    if max_lines is None:
        max_lines = float(_to_int(preset_defaults.get("max_lines"), 2))
    overrides["max_lines"] = max(1, int(max_lines))

    line_delay = _pick_numeric(cap_cfg, "lineDelay")
    if line_delay is None:
        line_delay = _as_number(preset_defaults.get("line_delay"))
    if line_delay is None:
        animation_defaults = preset_defaults.get("animation") or {}
        if isinstance(animation_defaults, dict):
            line_delay = _as_number(animation_defaults.get("delay_between_words"))
    overrides["line_delay"] = max(0.0, _to_float(line_delay, 0.0))

    font_size = cap_cfg.get("fontSize")
    if isinstance(font_size, (int, float)):
        overrides["font_size"] = int(font_size)

    if "italic" in cap_cfg:
        overrides["italic"] = bool(cap_cfg.get("italic"))

    if "underline" in cap_cfg:
        overrides["underline"] = bool(cap_cfg.get("underline"))

    font_family = cap_cfg.get("fontFamily")
    if isinstance(font_family, str) and font_family.strip():
        font_weight = cap_cfg.get("fontWeight")
        bold_hint = str(overrides.get("font_case") or "as_typed") == "uppercase"
        bold_override = _weight_implies_bold(_normalize_font_weight_token(font_weight))
        if bold_override is not None:
            overrides["bold"] = bold_override
            bold_hint = bold_override
        overrides["font_name"] = _resolve_ass_font_name(
            font_family=font_family,
            font_weight=font_weight,
            bold_hint=bold_hint,
        )
    if rtl_transcript:
        overrides["font_name"] = _default_rtl_caption_font_name()

    font_color = cap_cfg.get("fontColor")
    if isinstance(font_color, str) and font_color.strip():
        overrides["primary_color"] = to_ass_color(font_color)

    highlight_color = cap_cfg.get("highlightColor")
    if isinstance(highlight_color, str) and highlight_color.strip():
        overrides["highlight_color"] = to_ass_color(highlight_color)
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
    normalized_position = (
        str(position).strip().lower()
        if isinstance(position, str) and position.strip()
        else ""
    )
    if normalized_position == "custom":
        custom_y = _clamp(_to_int(cap_cfg.get("customY"), 0), 0, max(0, canvas_h - 1))
        overrides["position"] = "auto"
        overrides["alignment"] = 8
        overrides["margin_v"] = custom_y
        overrides["safe_margin_y"] = custom_y
    elif normalized_position:
        overrides["position"] = normalized_position

    if "backgroundBox" in cap_cfg:
        overrides["background_box"] = bool(cap_cfg.get("backgroundBox"))

    animation = cap_cfg.get("animation")
    if isinstance(animation, str) and animation.strip():
        overrides["animation"] = animation.strip()

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

    requested_style = resolve_caption_style_mode(cap_cfg)
    effective_style = resolve_rtl_safe_caption_style(requested_style, transcript)
    if effective_style != requested_style:
        logger.info(
            "[%s] Forcing grouped captions for RTL transcript (requested=%s effective=%s)",
            job_id,
            requested_style,
            effective_style,
        )

    overrides = _overrides_from_layout(
        cap_cfg,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        preset_name=preset_name,
        transcript=transcript,
    )
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
            if requested_position == "top":
                overrides["safe_margin_y"] = anchored_margin
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

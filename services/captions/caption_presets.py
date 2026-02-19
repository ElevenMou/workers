"""Preset catalog and normalization helpers for ASS caption rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict

AnimationType = Literal["fade", "slide_up", "pop", "karaoke", "none"]


class CaptionAnimation(TypedDict):
    type: AnimationType
    duration: float


class CaptionPreset(TypedDict, total=False):
    id: str
    label: str
    description: str
    font_name: str
    font_size: int
    primary_color: str
    secondary_color: str
    outline_color: str
    back_color: str
    alignment: int
    margin_v: int
    margin_l: int
    margin_r: int
    outline: int
    shadow: int
    bold: bool
    italic: bool
    underline: bool
    animation: CaptionAnimation
    word_highlight: bool
    max_chars_per_line: int
    max_lines: int
    safe_margin_x: int
    safe_margin_y: int
    position: str
    uppercase: bool
    punctuation_cleanup: bool
    background_box: bool
    line_delay: float


class CaptionTemplate(TypedDict, total=False):
    show: bool
    presetName: str
    animation: str
    position: str
    fontSize: int
    fontFamily: str
    fontColor: str
    highlightColor: str
    strokeColor: str
    shadowColor: str
    maxCharsPerCaption: int
    maxLines: int
    lineDelay: float
    uppercase: bool
    wordHighlight: bool
    backgroundBox: bool


ANIMATION_OPTIONS: tuple[AnimationType, ...] = (
    "none",
    "fade",
    "slide_up",
    "pop",
    "karaoke",
)

ANIMATION_ALIASES: dict[str, AnimationType] = {
    "none": "none",
    "off": "none",
    "no_animation": "none",
    "fade": "fade",
    "fade_in": "fade",
    "slide": "slide_up",
    "slide_up": "slide_up",
    "slideup": "slide_up",
    "pop": "pop",
    "bounce": "pop",
    "karaoke": "karaoke",
    "word_highlight": "karaoke",
}

CAPTION_STYLE_ALIASES: dict[str, str] = {
    # New style ids
    "clean_minimal": "clean_minimal",
    "bold_tiktok": "bold_tiktok",
    "word_highlighted": "word_highlighted",
    "cinematic_lower_third": "cinematic_lower_third",
    "block_background": "block_background",
    # Legacy mappings
    "animated": "word_highlighted",
    "grouped": "clean_minimal",
    "word_by_word": "word_highlighted",
    "progressive": "bold_tiktok",
    "punctuated": "clean_minimal",
    "static": "clean_minimal",
    "two_line": "cinematic_lower_third",
    "uppercase": "bold_tiktok",
    "lowercase": "clean_minimal",
    "headline": "cinematic_lower_third",
    "karaoke": "word_highlighted",
    "classic": "clean_minimal",
    "minimal": "clean_minimal",
    "focus_word": "word_highlighted",
    "build": "bold_tiktok",
    "split": "cinematic_lower_third",
    "sentence": "clean_minimal",
    "caps": "bold_tiktok",
    "lower": "clean_minimal",
    "title_case": "cinematic_lower_third",
}

CAPTION_TEMPLATE_DEFAULTS: CaptionTemplate = {
    "show": False,
    "presetName": "clean_minimal",
    "animation": "none",
    "position": "auto",
    "fontSize": 68,
    "fontFamily": "Montserrat-Bold",
    "fontColor": "&H00FFFFFF",
    "highlightColor": "&H0000C8FF",
    "strokeColor": "&H00000000",
    "shadowColor": "&H80000000",
    "maxCharsPerCaption": 28,
    "maxLines": 2,
    "lineDelay": 0.0,
    "uppercase": False,
    "wordHighlight": False,
    "backgroundBox": False,
}

CAPTION_PRESETS: dict[str, CaptionPreset] = {
    "clean_minimal": {
        "id": "clean_minimal",
        "label": "Clean Minimal",
        "description": "Simple high-contrast captions with subtle fade timing.",
        "font_name": "Montserrat-Bold",
        "font_size": 68,
        "primary_color": "&H00FFFFFF",
        "secondary_color": "&H0000C8FF",
        "outline_color": "&H00000000",
        "back_color": "&H70000000",
        "alignment": 2,
        "margin_v": 90,
        "margin_l": 56,
        "margin_r": 56,
        "outline": 3,
        "shadow": 1,
        "bold": True,
        "italic": False,
        "underline": False,
        "animation": {"type": "fade", "duration": 0.20},
        "word_highlight": False,
        "max_chars_per_line": 30,
        "max_lines": 2,
        "safe_margin_x": 56,
        "safe_margin_y": 90,
        "position": "auto",
        "uppercase": False,
        "punctuation_cleanup": True,
        "background_box": False,
        "line_delay": 0.00,
    },
    "bold_tiktok": {
        "id": "bold_tiktok",
        "label": "Bold TikTok",
        "description": "Heavy uppercase style with energetic pop animation.",
        "font_name": "Montserrat-ExtraBold",
        "font_size": 76,
        "primary_color": "&H00FFFFFF",
        "secondary_color": "&H0000A5FF",
        "outline_color": "&H00000000",
        "back_color": "&H70000000",
        "alignment": 2,
        "margin_v": 105,
        "margin_l": 48,
        "margin_r": 48,
        "outline": 4,
        "shadow": 2,
        "bold": True,
        "italic": False,
        "underline": False,
        "animation": {"type": "pop", "duration": 0.22},
        "word_highlight": False,
        "max_chars_per_line": 24,
        "max_lines": 2,
        "safe_margin_x": 48,
        "safe_margin_y": 105,
        "position": "auto",
        "uppercase": True,
        "punctuation_cleanup": True,
        "background_box": False,
        "line_delay": 0.03,
    },
    "word_highlighted": {
        "id": "word_highlighted",
        "label": "Word Highlighted",
        "description": "Karaoke-style per-word emphasis using ASS \\k timing.",
        "font_name": "Montserrat-Bold",
        "font_size": 72,
        "primary_color": "&H00FFFFFF",
        "secondary_color": "&H0000C8FF",
        "outline_color": "&H00000000",
        "back_color": "&H68000000",
        "alignment": 2,
        "margin_v": 96,
        "margin_l": 56,
        "margin_r": 56,
        "outline": 3,
        "shadow": 1,
        "bold": True,
        "italic": False,
        "underline": False,
        "animation": {"type": "karaoke", "duration": 0.30},
        "word_highlight": True,
        "max_chars_per_line": 26,
        "max_lines": 2,
        "safe_margin_x": 56,
        "safe_margin_y": 96,
        "position": "auto",
        "uppercase": False,
        "punctuation_cleanup": True,
        "background_box": False,
        "line_delay": 0.00,
    },
    "cinematic_lower_third": {
        "id": "cinematic_lower_third",
        "label": "Cinematic Lower Third",
        "description": "Lower-third presentation with soft slide-up motion.",
        "font_name": "Montserrat-SemiBold",
        "font_size": 62,
        "primary_color": "&H00F4F4F4",
        "secondary_color": "&H00B8E6FF",
        "outline_color": "&H00101010",
        "back_color": "&H90000000",
        "alignment": 1,
        "margin_v": 72,
        "margin_l": 88,
        "margin_r": 56,
        "outline": 2,
        "shadow": 0,
        "bold": True,
        "italic": False,
        "underline": False,
        "animation": {"type": "slide_up", "duration": 0.28},
        "word_highlight": False,
        "max_chars_per_line": 34,
        "max_lines": 2,
        "safe_margin_x": 72,
        "safe_margin_y": 72,
        "position": "bottom",
        "uppercase": False,
        "punctuation_cleanup": True,
        "background_box": False,
        "line_delay": 0.02,
    },
    "block_background": {
        "id": "block_background",
        "label": "Block Background",
        "description": "Text over opaque caption block for maximum readability.",
        "font_name": "Montserrat-Bold",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",
        "secondary_color": "&H0000FFFF",
        "outline_color": "&H00000000",
        "back_color": "&HC0222222",
        "alignment": 2,
        "margin_v": 86,
        "margin_l": 54,
        "margin_r": 54,
        "outline": 0,
        "shadow": 0,
        "bold": True,
        "italic": False,
        "underline": False,
        "animation": {"type": "fade", "duration": 0.18},
        "word_highlight": False,
        "max_chars_per_line": 28,
        "max_lines": 2,
        "safe_margin_x": 54,
        "safe_margin_y": 86,
        "position": "auto",
        "uppercase": False,
        "punctuation_cleanup": True,
        "background_box": True,
        "line_delay": 0.00,
    },
}

SUPPORTED_CAPTION_STYLES: set[str] = set(CAPTION_PRESETS)


def _normalize_animation(value: str | None) -> AnimationType:
    if not value:
        return "none"
    key = value.strip().lower().replace("-", "_")
    return ANIMATION_ALIASES.get(key, "none")


def normalize_caption_style(style: str | None) -> str:
    if not style:
        return CAPTION_TEMPLATE_DEFAULTS["presetName"]
    key = style.strip().lower()
    return CAPTION_STYLE_ALIASES.get(key, CAPTION_TEMPLATE_DEFAULTS["presetName"])


def list_supported_styles() -> list[str]:
    return sorted(SUPPORTED_CAPTION_STYLES)


def list_animation_presets() -> list[str]:
    return list(ANIMATION_OPTIONS)


def _normalize_override_key(key: str) -> str:
    return {
        "fontName": "font_name",
        "fontSize": "font_size",
        "primaryColor": "primary_color",
        "secondaryColor": "secondary_color",
        "outlineColor": "outline_color",
        "backColor": "back_color",
        "marginV": "margin_v",
        "marginL": "margin_l",
        "marginR": "margin_r",
        "wordHighlight": "word_highlight",
        "maxCharsPerLine": "max_chars_per_line",
        "maxCharsPerCaption": "max_chars_per_line",
        "maxLines": "max_lines",
        "safeMarginX": "safe_margin_x",
        "safeMarginY": "safe_margin_y",
        "backgroundBox": "background_box",
        "lineDelay": "line_delay",
    }.get(key, key)


def resolve_preset(
    preset_name: str | None,
    overrides: dict[str, Any] | None = None,
) -> CaptionPreset:
    key = normalize_caption_style(preset_name)
    base = CAPTION_PRESETS.get(key) or CAPTION_PRESETS[CAPTION_TEMPLATE_DEFAULTS["presetName"]]
    resolved: CaptionPreset = deepcopy(base)
    if not overrides:
        return resolved

    for raw_key, value in overrides.items():
        key_name = _normalize_override_key(str(raw_key))
        if key_name == "animation":
            if isinstance(value, dict):
                animation_type = _normalize_animation(str(value.get("type") or "none"))
                try:
                    duration = max(0.0, float(value.get("duration", 0.0)))
                except (TypeError, ValueError):
                    duration = 0.0
                resolved["animation"] = {"type": animation_type, "duration": duration}
                continue
            resolved["animation"] = {
                "type": _normalize_animation(str(value)),
                "duration": float((resolved.get("animation") or {}).get("duration", 0.0)),
            }
            continue
        resolved[key_name] = value  # type: ignore[index]
    return resolved


def resolve_caption_preset(preset: str | None) -> dict[str, Any]:
    """Return a UI-facing caption template derived from the given preset id."""
    cfg = resolve_preset(preset)
    animation = cfg.get("animation") or {"type": "none", "duration": 0.0}
    return {
        **CAPTION_TEMPLATE_DEFAULTS,
        "presetName": cfg.get("id", CAPTION_TEMPLATE_DEFAULTS["presetName"]),
        "animation": animation.get("type", "none"),
        "position": cfg.get("position", "auto"),
        "fontSize": int(cfg.get("font_size", CAPTION_TEMPLATE_DEFAULTS["fontSize"])),
        "fontFamily": str(cfg.get("font_name", CAPTION_TEMPLATE_DEFAULTS["fontFamily"])),
        "fontColor": str(cfg.get("primary_color", CAPTION_TEMPLATE_DEFAULTS["fontColor"])),
        "highlightColor": str(
            cfg.get("secondary_color", CAPTION_TEMPLATE_DEFAULTS["highlightColor"])
        ),
        "strokeColor": str(cfg.get("outline_color", CAPTION_TEMPLATE_DEFAULTS["strokeColor"])),
        "shadowColor": str(cfg.get("back_color", CAPTION_TEMPLATE_DEFAULTS["shadowColor"])),
        "maxCharsPerCaption": int(
            cfg.get("max_chars_per_line", CAPTION_TEMPLATE_DEFAULTS["maxCharsPerCaption"])
        ),
        "maxLines": int(cfg.get("max_lines", CAPTION_TEMPLATE_DEFAULTS["maxLines"])),
        "lineDelay": float(cfg.get("line_delay", CAPTION_TEMPLATE_DEFAULTS["lineDelay"])),
        "uppercase": bool(cfg.get("uppercase", CAPTION_TEMPLATE_DEFAULTS["uppercase"])),
        "wordHighlight": bool(
            cfg.get("word_highlight", CAPTION_TEMPLATE_DEFAULTS["wordHighlight"])
        ),
        "backgroundBox": bool(
            cfg.get("background_box", CAPTION_TEMPLATE_DEFAULTS["backgroundBox"])
        ),
    }


def list_caption_presets() -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = []
    for key in sorted(CAPTION_PRESETS):
        preset = CAPTION_PRESETS[key]
        presets.append(
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset.get("description", ""),
                "captions": resolve_caption_preset(preset["id"]),
                "style": {
                    "font_name": preset.get("font_name"),
                    "font_size": preset.get("font_size"),
                    "primary_color": preset.get("primary_color"),
                    "outline_color": preset.get("outline_color"),
                    "back_color": preset.get("back_color"),
                    "alignment": preset.get("alignment"),
                    "margin_v": preset.get("margin_v"),
                    "animation": preset.get("animation"),
                    "word_highlight": preset.get("word_highlight"),
                },
            }
        )
    return presets


def _hex_to_ass(hex_color: str) -> str:
    raw = hex_color.strip().lstrip("#")
    if len(raw) == 6:
        r, g, b = raw[0:2], raw[2:4], raw[4:6]
        return f"&H00{b}{g}{r}".upper()
    if len(raw) == 8:
        a, r, g, b = raw[0:2], raw[2:4], raw[4:6], raw[6:8]
        return f"&H{a}{b}{g}{r}".upper()
    return "&H00FFFFFF"


def to_ass_color(value: str | None, fallback: str = "&H00FFFFFF") -> str:
    """Best-effort conversion from named/css/ASS colors to ASS format."""
    if not value:
        return fallback
    token = value.strip()
    if token.startswith("&H"):
        return token.upper().rstrip("&")
    if token.startswith("#"):
        return _hex_to_ass(token)
    named: dict[str, str] = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "yellow": "&H0000FFFF",
        "cyan": "&H00FFFF00",
        "magenta": "&H00FF00FF",
        "orange": "&H0000A5FF",
        "gray": "&H00808080",
        "grey": "&H00808080",
    }
    return named.get(token.lower(), fallback)


__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "AnimationType",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "SUPPORTED_CAPTION_STYLES",
    "CaptionAnimation",
    "CaptionPreset",
    "CaptionTemplate",
    "_normalize_animation",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "resolve_caption_preset",
    "resolve_preset",
    "to_ass_color",
]

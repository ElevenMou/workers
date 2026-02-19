"""Compatibility exports for preset configuration (new preset architecture)."""

from services.captions.caption_presets import (
    ANIMATION_ALIASES,
    ANIMATION_OPTIONS,
    CAPTION_PRESETS,
    CAPTION_STYLE_ALIASES,
    CAPTION_TEMPLATE_DEFAULTS,
    SUPPORTED_CAPTION_STYLES,
    _normalize_animation,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    resolve_caption_preset,
    resolve_preset,
    to_ass_color,
)


def _normalize_font_case(font_case: str | None) -> str:
    key = (font_case or "as_typed").strip().lower()
    if key == "uppercase":
        return "uppercase"
    if key == "lowercase":
        return "lowercase"
    return "as_typed"


__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "SUPPORTED_CAPTION_STYLES",
    "_normalize_animation",
    "_normalize_font_case",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "resolve_caption_preset",
    "resolve_preset",
    "to_ass_color",
]

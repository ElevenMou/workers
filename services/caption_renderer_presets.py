"""Compatibility wrapper for caption presets module."""

from services.captions.presets import (
    ANIMATION_ALIASES,
    ANIMATION_OPTIONS,
    CAPTION_PRESETS,
    CAPTION_STYLE_ALIASES,
    CAPTION_TEMPLATE_DEFAULTS,
    SUPPORTED_CAPTION_STYLES,
    _normalize_animation,
    _normalize_font_case,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    resolve_caption_preset,
)

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
]

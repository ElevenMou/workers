"""Caption rendering feature package."""

from services.captions.presets import (
    ANIMATION_ALIASES,
    ANIMATION_OPTIONS,
    CAPTION_PRESETS,
    CAPTION_STYLE_ALIASES,
    CAPTION_TEMPLATE_DEFAULTS,
    SUPPORTED_CAPTION_STYLES,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    resolve_caption_preset,
)
from services.captions.render import render_ass
from services.captions.segments import extract_clip_segments

__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "SUPPORTED_CAPTION_STYLES",
    "extract_clip_segments",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "render_ass",
    "resolve_caption_preset",
]

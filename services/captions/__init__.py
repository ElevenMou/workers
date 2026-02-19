"""Caption rendering package (preset-driven ASS generation)."""

from services.captions.ass_generator import format_ass_timestamp, generate_ass_content, generate_ass_file
from services.captions.caption_presets import (
    ANIMATION_ALIASES,
    ANIMATION_OPTIONS,
    CAPTION_PRESETS,
    CAPTION_STYLE_ALIASES,
    CAPTION_TEMPLATE_DEFAULTS,
    PRESETS,
    SUPPORTED_CAPTION_STYLES,
    get_preset,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    resolve_caption_preset,
    resolve_preset,
    to_ass_color,
)
from services.captions.render import render_ass
from services.captions.renderer import render_captions, resolve_font_dir
from services.captions.segments import extract_clip_segments
from services.captions.positioning import compute_video_anchored_margin_v

__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "PRESETS",
    "SUPPORTED_CAPTION_STYLES",
    "get_preset",
    "extract_clip_segments",
    "format_ass_timestamp",
    "generate_ass_content",
    "generate_ass_file",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "compute_video_anchored_margin_v",
    "render_ass",
    "render_captions",
    "resolve_caption_preset",
    "resolve_font_dir",
    "resolve_preset",
    "to_ass_color",
]

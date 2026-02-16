"""Caption style normalization and preset access helpers."""

from typing import Any

from services.captions.preset_catalog import (
    ANIMATION_ALIASES,
    ANIMATION_OPTIONS,
    CAPTION_PRESETS,
    CAPTION_STYLE_ALIASES,
    CAPTION_TEMPLATE_DEFAULTS,
    SUPPORTED_CAPTION_STYLES,
    _FONT_CASES,
)
from services.models.caption import (
    CaptionPreset,
    CaptionPresetDefinition,
    CaptionTemplate,
)


def list_supported_styles() -> list[str]:
    """Return the list of supported caption styles."""
    return sorted(SUPPORTED_CAPTION_STYLES)


def list_animation_presets() -> list[str]:
    """Return animation names exposed to clients."""
    return list(ANIMATION_OPTIONS)


def _preset_to_caption_template(preset: CaptionPresetDefinition) -> CaptionTemplate:
    """Build explicit caption settings for a UI template preset."""
    preset_captions = preset.get("captions")
    if isinstance(preset_captions, dict):
        merged = {**CAPTION_TEMPLATE_DEFAULTS, **preset_captions}
        return {k: merged[k] for k in CAPTION_TEMPLATE_DEFAULTS}

    return {
        **CAPTION_TEMPLATE_DEFAULTS,
        "style": preset.get("style", CAPTION_TEMPLATE_DEFAULTS["style"]),
        "animation": preset.get("animation", CAPTION_TEMPLATE_DEFAULTS["animation"]),
        "position": preset.get("position", CAPTION_TEMPLATE_DEFAULTS["position"]),
        "linesPerPage": preset.get(
            "linesPerPage",
            CAPTION_TEMPLATE_DEFAULTS["linesPerPage"],
        ),
        "maxWordsPerLine": preset.get(
            "maxWordsPerLine",
            CAPTION_TEMPLATE_DEFAULTS["maxWordsPerLine"],
        ),
    }


def list_caption_presets() -> list[CaptionPreset]:
    """Return built-in caption templates with explicit caption config values."""
    presets: list[dict[str, Any]] = []
    for key in CAPTION_PRESETS:
        preset = CAPTION_PRESETS[key]
        presets.append(
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset.get("description", ""),
                "captions": _preset_to_caption_template(preset),
            }
        )
    return presets


def resolve_caption_preset(preset: str | None) -> dict[str, Any]:
    """Backward-compat helper: resolve preset id to explicit caption values."""
    if not preset:
        return {}
    preset_cfg = CAPTION_PRESETS.get(preset)
    if not preset_cfg:
        return {}
    return _preset_to_caption_template(preset_cfg)


def _normalize_animation(animation: str | None) -> str:
    if not animation:
        return "none"
    key = animation.strip().lower().replace("-", "_")
    return ANIMATION_ALIASES.get(key, "none")


def _normalize_font_case(font_case: str | None) -> str:
    key = (font_case or "as_typed").strip().lower()
    if key not in _FONT_CASES:
        return "as_typed"
    return key


def normalize_caption_style(style: str | None) -> str:
    """Normalize a user-provided style to a supported style name."""
    if style is None:
        return "animated"

    key = style.strip().lower()
    if not key:
        return "animated"

    normalized = CAPTION_STYLE_ALIASES.get(key, key)
    if normalized not in SUPPORTED_CAPTION_STYLES:
        return "static"
    return normalized


__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "SUPPORTED_CAPTION_STYLES",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "resolve_caption_preset",
]

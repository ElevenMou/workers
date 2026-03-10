"""Constants and helpers for clip generation canvas/layout options."""

from services.clips.models import QualityPreset

DEFAULT_CANVAS_ASPECT_RATIO = "9:16"
DEFAULT_VIDEO_SCALE_MODE = "fit"

CANVAS_ASPECT_RATIOS = ["9:16", "1:1", "4:5", "16:9"]
CANVAS_ASPECT_RATIO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "16:9": (1920, 1080),
}
CANVAS_ASPECT_RATIO_ALIASES = {
    "9:16": "9:16",
    "9x16": "9:16",
    "portrait": "9:16",
    "vertical": "9:16",
    "1:1": "1:1",
    "square": "1:1",
    "4:5": "4:5",
    "4x5": "4:5",
    "feed": "4:5",
    "16:9": "16:9",
    "16x9": "16:9",
    "landscape": "16:9",
    "horizontal": "16:9",
}
VIDEO_SCALE_MODES = ["fit", "fill"]
VIDEO_SCALE_MODE_ALIASES = {
    "fit": "fit",
    "contain": "fit",
    "fill": "fill",
    "cover": "fill",
}

TITLE_BAR_V_PAD = 16
TITLE_GAP = 12
TITLE_LINE_HEIGHT_RATIO = 0.85

QUALITY_PRESETS: dict[str, QualityPreset] = {
    "low": {"crf": 23, "preset": "fast"},
    "medium": {"crf": 18, "preset": "medium"},
    "high": {"crf": 15, "preset": "slow"},
}


def intermediate_quality_preset(base: QualityPreset) -> QualityPreset:
    """Return a higher-fidelity preset for intermediate transcodes.

    The generation pipeline can encode multiple times before the final output.
    Using a lower CRF for intermediates preserves detail and avoids cumulative
    quality loss, while the final pass still honors the selected output quality.
    """
    base_crf = int(base.get("crf", 23))
    # Keep intermediates visibly cleaner than the final CRF target.
    intermediate_crf = max(10, min(28, base_crf - 8))
    return {"crf": intermediate_crf, "preset": str(base.get("preset", "medium"))}


CHAR_WIDTH_RATIO = 0.52


def normalize_canvas_aspect_ratio(aspect_ratio: str | None) -> str:
    """Normalize a user-provided aspect ratio to a supported value."""
    if not aspect_ratio:
        return DEFAULT_CANVAS_ASPECT_RATIO
    key = aspect_ratio.strip().lower()
    return CANVAS_ASPECT_RATIO_ALIASES.get(key, DEFAULT_CANVAS_ASPECT_RATIO)


def canvas_size_for_aspect_ratio(aspect_ratio: str | None) -> tuple[int, int]:
    """Return concrete ``(width, height)`` pixels for a canvas aspect ratio."""
    normalized = normalize_canvas_aspect_ratio(aspect_ratio)
    return CANVAS_ASPECT_RATIO_DIMENSIONS[normalized]


def normalize_video_scale_mode(scale_mode: str | None) -> str:
    """Normalize fit/fill mode aliases to supported values."""
    if not scale_mode:
        return DEFAULT_VIDEO_SCALE_MODE
    key = scale_mode.strip().lower()
    return VIDEO_SCALE_MODE_ALIASES.get(key, DEFAULT_VIDEO_SCALE_MODE)

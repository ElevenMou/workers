"""Constants for clip generation on a 9:16 canvas."""

from services.clips.models import QualityPreset

CANVAS_W = 1080
CANVAS_H = 1920

TITLE_BAR_V_PAD = 16
TITLE_GAP = 12

QUALITY_PRESETS: dict[str, QualityPreset] = {
    "low": {"crf": 28, "preset": "veryfast"},
    "medium": {"crf": 23, "preset": "medium"},
    "high": {"crf": 18, "preset": "slow"},
}

CHAR_WIDTH_RATIO = 0.52

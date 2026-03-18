"""Clip generator model types."""

from typing import TypedDict


class _QualityPresetRequired(TypedDict):
    crf: int
    preset: str
    resolution: int


class QualityPreset(_QualityPresetRequired, total=False):
    profile: str        # H.264 profile, e.g. "high"
    level: str          # H.264 level, e.g. "4.1"
    tune: str           # x264 tune, e.g. "film"
    maxrate: str        # VBV max bitrate, e.g. "12M"
    bufsize: str        # VBV buffer size, e.g. "24M"
    audio_bitrate: str  # Override default "256k"
    fps: int | None     # None = preserve source fps; absent = default 30


class ClipLayout(TypedDict):
    canvas_w: int
    canvas_h: int
    vid_w: int
    vid_h: int
    vid_x: int
    vid_y: int
    title_padding_x: int
    title_bar_x: int
    title_bar_w: int
    title_bar_y: int
    title_bar_h: int
    title_text_y: int


class ClipGenerationResult(TypedDict):
    clip_path: str
    file_size: int
    master_path: str
    master_file_size: int
    delivery_path: str
    delivery_file_size: int
    delivery_profile: str
    intermediates: list[str]

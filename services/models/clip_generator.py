"""Clip generator model types."""

from typing import TypedDict


class QualityPreset(TypedDict):
    crf: int
    preset: str


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
    intermediates: list[str]

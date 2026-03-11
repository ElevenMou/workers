"""Shared caption model types (preset-driven ASS rendering)."""

from __future__ import annotations

from typing import Literal, TypedDict

AnimationType = Literal["none", "fade", "pop", "slide_up", "bounce", "glow"]


class CaptionAnimation(TypedDict):
    type: AnimationType
    duration: float
    delay_between_words: float


class CaptionTemplate(TypedDict, total=False):
    show: bool
    presetName: str
    animation: AnimationType
    position: str
    style: str
    fontSize: int
    fontFamily: str
    fontWeight: str
    fontCase: str
    fontColor: str
    highlightColor: str
    strokeColor: str
    strokeThickness: int
    shadowColor: str
    shadowX: int
    shadowY: int
    shadowBlur: int
    italic: bool
    underline: bool
    linesPerPage: int
    maxWordsPerLine: int
    maxCharsPerCaption: int
    maxLines: int
    lineDelay: float
    uppercase: bool
    wordHighlight: bool
    backgroundBox: bool


class CaptionTemplateOverrides(CaptionTemplate, total=False):
    pass


class CaptionPreset(TypedDict):
    id: str
    label: str
    description: str
    captions: CaptionTemplate


class CaptionPresetDefinition(TypedDict, total=False):
    id: str
    label: str
    description: str
    captions: CaptionTemplateOverrides


__all__ = [
    "AnimationType",
    "CaptionAnimation",
    "CaptionPreset",
    "CaptionPresetDefinition",
    "CaptionTemplate",
    "CaptionTemplateOverrides",
]

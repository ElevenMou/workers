"""Shared caption model types (preset-driven ASS rendering)."""

from __future__ import annotations

from typing import Literal, TypedDict

AnimationType = Literal["fade", "slide_up", "pop", "karaoke", "none"]


class CaptionAnimation(TypedDict):
    type: AnimationType
    duration: float


class CaptionTemplate(TypedDict, total=False):
    show: bool
    presetName: str
    animation: AnimationType
    position: str
    fontSize: int
    fontFamily: str
    fontColor: str
    highlightColor: str
    strokeColor: str
    shadowColor: str
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

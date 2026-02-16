"""Caption renderer model types."""

from typing import Literal, TypedDict

CaptionStyle = Literal[
    "animated",
    "grouped",
    "headline",
    "lowercase",
    "progressive",
    "punctuated",
    "static",
    "two_line",
    "uppercase",
    "word_by_word",
]
FontCase = Literal["as_typed", "uppercase", "lowercase", "headline"]


class CaptionTemplate(TypedDict):
    show: bool
    style: CaptionStyle
    animation: str
    fontSize: int
    fontColor: str
    fontFamily: str
    fontWeight: str
    fontCase: FontCase
    italic: bool
    underline: bool
    strokeColor: str
    strokeThickness: int
    shadowColor: str
    shadowX: int
    shadowY: int
    shadowBlur: int
    highlightColor: str
    position: str
    linesPerPage: int
    maxWordsPerLine: int
    maxCharsPerCaption: int


class CaptionTemplateOverrides(TypedDict, total=False):
    show: bool
    style: CaptionStyle
    animation: str
    fontSize: int
    fontColor: str
    fontFamily: str
    fontWeight: str
    fontCase: FontCase
    italic: bool
    underline: bool
    strokeColor: str
    strokeThickness: int
    shadowColor: str
    shadowX: int
    shadowY: int
    shadowBlur: int
    highlightColor: str
    position: str
    linesPerPage: int
    maxWordsPerLine: int
    maxCharsPerCaption: int


class CaptionPreset(TypedDict):
    id: str
    label: str
    description: str
    captions: CaptionTemplate


class CaptionPresetDefinition(TypedDict, total=False):
    id: str
    label: str
    description: str
    style: CaptionStyle
    animation: str
    position: str
    linesPerPage: int
    maxWordsPerLine: int
    captions: CaptionTemplateOverrides


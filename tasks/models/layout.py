"""Task-side layout model types and defaults."""

from typing import Any, TypedDict


class VideoLayout(TypedDict, total=False):
    widthPct: int
    positionY: str
    canvasAspectRatio: str
    videoScaleMode: str


class TitleLayout(TypedDict, total=False):
    show: bool
    fontSize: int
    fontColor: str
    fontFamily: str
    align: str
    strokeWidth: int
    strokeColor: str
    barEnabled: bool
    barColor: str
    paddingX: int
    positionY: str


class CaptionLayout(TypedDict, total=False):
    show: bool
    presetName: str
    style: str
    animation: str
    fontSize: int
    fontColor: str
    fontFamily: str
    fontWeight: str
    fontCase: str
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
    maxLines: int
    lineDelay: float
    uppercase: bool
    wordHighlight: bool
    backgroundBox: bool


DEFAULT_VIDEO_LAYOUT: VideoLayout = {
    "widthPct": 100,
    "positionY": "middle",
    "canvasAspectRatio": "9:16",
    "videoScaleMode": "fit",
}
DEFAULT_TITLE_LAYOUT: TitleLayout = {
    "show": True,
    "fontSize": 48,
    "fontColor": "white",
    "fontFamily": "",
    "align": "left",
    "strokeWidth": 0,
    "strokeColor": "black",
    "barEnabled": True,
    "barColor": "black@0.5",
    "paddingX": 16,
    "positionY": "above_video",
}
DEFAULT_CAPTION_LAYOUT: CaptionLayout = {
    "show": False,
    "presetName": "clean_minimal",
    "style": "clean_minimal",
    "animation": "none",
    "fontSize": 68,
    "fontColor": "&H00FFFFFF",
    "fontFamily": "Montserrat-Bold",
    "fontWeight": "bold",
    "fontCase": "as_typed",
    "italic": False,
    "underline": False,
    "strokeColor": "&H00000000",
    "strokeThickness": 3,
    "shadowColor": "&H80000000",
    "shadowX": 1,
    "shadowY": 1,
    "shadowBlur": 1,
    "highlightColor": "&H0000C8FF",
    "position": "auto",
    "linesPerPage": 2,
    "maxWordsPerLine": 5,
    "maxCharsPerCaption": 30,
    "maxLines": 2,
    "lineDelay": 0.0,
    "uppercase": False,
    "wordHighlight": False,
    "backgroundBox": False,
}


def merge_layout_configs(
    video_cfg: dict[str, Any] | None,
    title_cfg: dict[str, Any] | None,
    caption_cfg: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Merge user layout overrides with task defaults."""
    return (
        {**DEFAULT_VIDEO_LAYOUT, **(video_cfg or {})},
        {**DEFAULT_TITLE_LAYOUT, **(title_cfg or {})},
        {**DEFAULT_CAPTION_LAYOUT, **(caption_cfg or {})},
    )

"""Task-side layout model types and defaults."""

from typing import Any, TypedDict


class VideoLayout(TypedDict, total=False):
    widthPct: int
    positionY: str
    customX: int
    customY: int
    customWidth: int
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
    customX: int
    customY: int
    customWidth: int


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
    customX: int
    customY: int
    customWidth: int


DEFAULT_VIDEO_LAYOUT: VideoLayout = {
    "widthPct": 100,
    "positionY": "middle",
    "customX": 0,
    "customY": 0,
    "customWidth": 1080,
    "canvasAspectRatio": "9:16",
    "videoScaleMode": "fit",
}
DEFAULT_TITLE_LAYOUT: TitleLayout = {
    "show": True,
    "fontSize": 48,
    "fontColor": "#FFFFFF",
    "fontFamily": "",
    "align": "left",
    "strokeWidth": 0,
    "strokeColor": "#000000",
    "barEnabled": True,
    "barColor": "#000000",
    "paddingX": 16,
    "positionY": "above_video",
    "customX": 0,
    "customY": 0,
    "customWidth": 1080,
}
DEFAULT_CAPTION_LAYOUT: CaptionLayout = {
    "show": False,
    "presetName": "clean",
    "style": "grouped",
    "animation": "fade",
    "fontSize": 44,
    "fontColor": "#FFFFFF",
    "fontFamily": "Montserrat",
    "fontWeight": "bold",
    "fontCase": "as_typed",
    "italic": False,
    "underline": False,
    "strokeColor": "#000000",
    "strokeThickness": 6,
    "shadowColor": "#000000",
    "shadowX": 2,
    "shadowY": 2,
    "shadowBlur": 2,
    "highlightColor": "#FFD700",
    "position": "bottom",
    "linesPerPage": 2,
    "maxWordsPerLine": 4,
    "maxCharsPerCaption": 42,
    "maxLines": 2,
    "lineDelay": 0.05,
    "uppercase": False,
    "wordHighlight": False,
    "backgroundBox": False,
    "customX": 44,
    "customY": 1460,
    "customWidth": 992,
}


class IntroOutroLayout(TypedDict, total=False):
    enabled: bool
    type: str  # "video" or "image"
    storagePath: str
    durationSeconds: float  # used when type == "image"


class OverlayLayout(TypedDict, total=False):
    enabled: bool
    storagePath: str
    widthPx: int
    x: int
    y: int


DEFAULT_INTRO_LAYOUT: IntroOutroLayout = {
    "enabled": False,
    "type": "image",
    "storagePath": "",
    "durationSeconds": 3.0,
}
DEFAULT_OUTRO_LAYOUT: IntroOutroLayout = {
    "enabled": False,
    "type": "image",
    "storagePath": "",
    "durationSeconds": 3.0,
}
DEFAULT_OVERLAY_LAYOUT: OverlayLayout = {
    "enabled": False,
    "storagePath": "",
    "widthPx": 200,
    "x": 0,
    "y": 0,
}


def merge_layout_configs(
    video_cfg: dict[str, Any] | None,
    title_cfg: dict[str, Any] | None,
    caption_cfg: dict[str, Any] | None,
    intro_cfg: dict[str, Any] | None = None,
    outro_cfg: dict[str, Any] | None = None,
    overlay_cfg: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Merge user layout overrides with task defaults."""
    return (
        {**DEFAULT_VIDEO_LAYOUT, **(video_cfg or {})},
        {**DEFAULT_TITLE_LAYOUT, **(title_cfg or {})},
        {**DEFAULT_CAPTION_LAYOUT, **(caption_cfg or {})},
        {**DEFAULT_INTRO_LAYOUT, **(intro_cfg or {})},
        {**DEFAULT_OUTRO_LAYOUT, **(outro_cfg or {})},
        {**DEFAULT_OVERLAY_LAYOUT, **(overlay_cfg or {})},
    )

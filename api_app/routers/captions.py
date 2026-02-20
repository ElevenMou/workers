"""Caption options/presets endpoints."""

from fastapi import APIRouter

from api_app.constants import (
    CAPTION_ANIMATIONS,
    CAPTION_FONT_CASES,
    CAPTION_LINES_PER_PAGE_OPTIONS,
    CAPTION_POSITIONS,
    CAPTION_STYLES,
)
from api_app.models import (
    CaptionModesResponse,
    CaptionOptionsResponse,
    CaptionPresetsResponse,
)
from services.caption_renderer import (
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
)

router = APIRouter()


@router.get("/captions/modes", response_model=CaptionModesResponse)
def caption_modes() -> CaptionModesResponse:
    """Return all supported caption generation modes."""
    return CaptionModesResponse(modes=list_supported_styles())


@router.get("/captions/options", response_model=CaptionOptionsResponse)
def caption_options() -> CaptionOptionsResponse:
    """Return caption options for CapCut-like UI controls."""
    return CaptionOptionsResponse(
        styles=CAPTION_STYLES,
        animations=CAPTION_ANIMATIONS,
        presets=list_caption_presets(),
        fontCases=CAPTION_FONT_CASES,
        positions=CAPTION_POSITIONS,
        linesPerPageOptions=CAPTION_LINES_PER_PAGE_OPTIONS,
    )


@router.get("/captions/presets", response_model=CaptionPresetsResponse)
def caption_presets() -> CaptionPresetsResponse:
    """Return caption templates with explicit caption payload values."""
    return CaptionPresetsResponse(presets=list_caption_presets())

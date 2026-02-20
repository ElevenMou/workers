"""Caption preset catalog and normalization helpers for ASS rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

AnimationType = Literal["none", "fade", "pop", "slide_up", "bounce", "glow", "karaoke"]


class CaptionAnimation(TypedDict):
    type: AnimationType
    duration: float
    delay_between_words: float


class CaptionTemplate(TypedDict, total=False):
    show: bool
    presetName: str
    animation: str
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
    style: str


class CaptionPreset:
    """Preset definition used by ASS generator and frontend style selector."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        font_name: str = "Montserrat-Bold",
        font_size: int = 72,
        primary_color: str = "&H00FFFFFF",
        secondary_color: str = "&H000000FF",
        outline_color: str = "&H00000000",
        back_color: str = "&H80000000",
        bold: bool = True,
        italic: bool = False,
        outline_size: int = 3,
        shadow_size: int = 0,
        alignment: int = 2,
        margin_v: int = 80,
        margin_h: int = 60,
        position: str = "bottom",
        max_words_per_line: int = 4,
        animation: Mapping[str, Any] | None = None,
        word_highlight: bool = False,
        highlight_color: str = "&H0000FFFF",
        background_box: bool = False,
        background_padding: int = 15,
        uppercase: bool = False,
        safe_area: bool = True,
        description: str = "",
        style: str = "grouped",
    ):
        self.name = str(name).strip().lower()
        self.display_name = str(display_name).strip()
        self.font_name = str(font_name).strip() or "Montserrat-Bold"
        self.font_size = max(10, int(font_size))
        self.primary_color = str(primary_color)
        self.secondary_color = str(secondary_color)
        self.outline_color = str(outline_color)
        self.back_color = str(back_color)
        self.bold = bool(bold)
        self.italic = bool(italic)
        self.outline_size = max(0, int(outline_size))
        self.shadow_size = max(0, int(shadow_size))
        self.alignment = max(1, min(9, int(alignment)))
        self.margin_v = max(0, int(margin_v))
        self.margin_h = max(0, int(margin_h))
        normalized_position = str(position).strip().lower()
        if normalized_position not in {"auto", "top", "middle", "bottom"}:
            normalized_position = "bottom"
        self.position = normalized_position
        self.max_words_per_line = max(1, int(max_words_per_line))
        self.animation: CaptionAnimation = _normalize_animation_config(animation)
        self.word_highlight = bool(word_highlight)
        self.highlight_color = str(highlight_color)
        self.background_box = bool(background_box)
        self.background_padding = max(0, int(background_padding))
        self.uppercase = bool(uppercase)
        self.safe_area = bool(safe_area)
        self.description = str(description)
        normalized_style = str(style).strip().lower()
        if normalized_style not in {"grouped", "word_by_word", "karaoke"}:
            normalized_style = "grouped"
        self.style = normalized_style

    def to_style_dict(self) -> dict[str, Any]:
        """Return normalized style config used by ASS generator."""
        max_chars_per_line = max(12, int(self.max_words_per_line * 6))
        safe_margin_x = self.margin_h if self.safe_area else 0
        safe_margin_y = self.margin_v if self.safe_area else 0
        highlight = to_ass_color(self.highlight_color, fallback="&H0000FFFF")
        secondary = highlight if self.word_highlight else to_ass_color(self.secondary_color, highlight)

        return {
            "id": self.name,
            "label": self.display_name,
            "description": self.description,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "primary_color": to_ass_color(self.primary_color),
            "secondary_color": secondary,
            "outline_color": to_ass_color(self.outline_color, fallback="&H00000000"),
            "back_color": to_ass_color(self.back_color, fallback="&H80000000"),
            "bold": self.bold,
            "italic": self.italic,
            "underline": False,
            "outline": self.outline_size,
            "shadow": self.shadow_size,
            "alignment": self.alignment,
            "margin_v": self.margin_v,
            "margin_l": self.margin_h,
            "margin_r": self.margin_h,
            "position": self.position,
            "max_words_per_line": self.max_words_per_line,
            "max_chars_per_line": max_chars_per_line,
            "max_lines": 2,
            "safe_margin_x": safe_margin_x,
            "safe_margin_y": safe_margin_y,
            "uppercase": self.uppercase,
            "punctuation_cleanup": True,
            "background_box": self.background_box,
            "background_padding": self.background_padding,
            "line_delay": float(self.animation.get("delay_between_words", 0.0)),
            "animation": deepcopy(self.animation),
            "word_highlight": self.word_highlight,
            "highlight_color": highlight,
            "safe_area": self.safe_area,
            "style": self.style,
        }


ANIMATION_OPTIONS: tuple[AnimationType, ...] = (
    "none",
    "fade",
    "pop",
    "slide_up",
    "bounce",
    "glow",
    "karaoke",
)

ANIMATION_ALIASES: dict[str, AnimationType] = {
    "none": "none",
    "off": "none",
    "no_animation": "none",
    "fade": "fade",
    "fade_in": "fade",
    "pop": "pop",
    "punch": "pop",
    "scale": "pop",
    "zoom": "pop",
    "zoom_in": "pop",
    "slide": "slide_up",
    "slide_up": "slide_up",
    "slideup": "slide_up",
    "bounce": "bounce",
    "glow": "glow",
    "neon": "glow",
    "karaoke": "karaoke",
    "word_highlight": "karaoke",
}

_WHITE = "&H00FFFFFF"
_BLACK = "&H00000000"
_SEMI_BLACK = "&H80000000"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_animation(value: str | None) -> AnimationType:
    if not value:
        return "none"
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return ANIMATION_ALIASES.get(key, "none")


def _normalize_animation_config(animation: Mapping[str, Any] | None) -> CaptionAnimation:
    payload = dict(animation or {})
    animation_type = _normalize_animation(str(payload.get("type") or "none"))
    duration_default = 0.0 if animation_type == "none" else 0.2
    return {
        "type": animation_type,
        "duration": max(0.0, _as_float(payload.get("duration"), duration_default)),
        "delay_between_words": max(
            0.0,
            _as_float(payload.get("delay_between_words"), 0.05),
        ),
    }


# ---------------------------------------------------------------------------
# 20 new presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, CaptionPreset] = {
    "hormozi": CaptionPreset(
        name="hormozi",
        display_name="Hormozi",
        description="Business guru style with karaoke gold highlight.",
        font_name="Montserrat-Black",
        font_size=72,
        primary_color=_WHITE,
        secondary_color="&H0000D7FF",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=5,
        shadow_size=2,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color="&H0000D7FF",
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="karaoke",
    ),
    "mrbeast": CaptionPreset(
        name="mrbeast",
        display_name="MrBeast",
        description="Bold impact style, one word at a time.",
        font_name="Montserrat-Black",
        font_size=80,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=7,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=3,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="word_by_word",
    ),
    "clean": CaptionPreset(
        name="clean",
        display_name="Clean",
        description="Everyday minimal subtitle look with soft fade.",
        font_name="Montserrat-Bold",
        font_size=48,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=2,
        shadow_size=1,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "karaoke_gold": CaptionPreset(
        name="karaoke_gold",
        display_name="Karaoke Gold",
        description="Music/sweep karaoke style with gold highlight.",
        font_name="Montserrat-Bold",
        font_size=64,
        primary_color=_WHITE,
        secondary_color="&H0000D7FF",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=1,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "karaoke", "duration": 0.2, "delay_between_words": 0.08},
        word_highlight=True,
        highlight_color="&H0000D7FF",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="karaoke",
    ),
    "boxed": CaptionPreset(
        name="boxed",
        display_name="Boxed",
        description="News-box style captions with dark background slab.",
        font_name="Montserrat-Bold",
        font_size=52,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=0,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "slide_up", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=True,
        background_padding=15,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "neon": CaptionPreset(
        name="neon",
        display_name="Neon",
        description="Cyberpunk neon glow aesthetic.",
        font_name="Montserrat-Black",
        font_size=60,
        primary_color="&H00FFE500",
        secondary_color="&H00FFE500",
        outline_color="&H00330000",
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=2,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "glow", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color="&H00FFE500",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "cinematic": CaptionPreset(
        name="cinematic",
        display_name="Cinematic",
        description="Movie subtitle style with elegant serif font.",
        font_name="Playfair Display",
        font_size=44,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=1,
        shadow_size=2,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=6,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "street": CaptionPreset(
        name="street",
        display_name="Street",
        description="Urban bold uppercase with slide-up motion.",
        font_name="Montserrat-Black",
        font_size=68,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=6,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "slide_up", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="grouped",
    ),
    "pastel": CaptionPreset(
        name="pastel",
        display_name="Pastel",
        description="Soft aesthetic pink tones with gentle fade.",
        font_name="Montserrat",
        font_size=52,
        primary_color="&H00D0B4FF",
        secondary_color="&H00D0B4FF",
        outline_color=_WHITE,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=3,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color="&H00D0B4FF",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "fire": CaptionPreset(
        name="fire",
        display_name="Fire",
        description="Hot intense style with orange and dark red.",
        font_name="Montserrat-Black",
        font_size=64,
        primary_color="&H00006BFF",
        secondary_color="&H00006BFF",
        outline_color="&H0000008B",
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color="&H00006BFF",
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="grouped",
    ),
    "ice": CaptionPreset(
        name="ice",
        display_name="Ice",
        description="Cool calm light-blue tones with soft fade.",
        font_name="Montserrat",
        font_size=56,
        primary_color="&H00FFE4B4",
        secondary_color="&H00FFE4B4",
        outline_color=_WHITE,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=2,
        shadow_size=1,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color="&H00FFE4B4",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "retro": CaptionPreset(
        name="retro",
        display_name="Retro",
        description="Terminal/hacker green-on-black monospace glow.",
        font_name="Space Mono",
        font_size=48,
        primary_color="&H0014FF39",
        secondary_color="&H0014FF39",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=0,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "glow", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color="&H0014FF39",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "news": CaptionPreset(
        name="news",
        display_name="News",
        description="Lower-third news style with background box.",
        font_name="Montserrat",
        font_size=44,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=0,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=6,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=True,
        background_padding=15,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "contrast": CaptionPreset(
        name="contrast",
        display_name="Contrast",
        description="Accessibility-first high-contrast heavy outline.",
        font_name="Montserrat-Black",
        font_size=72,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=8,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="grouped",
    ),
    "podcast": CaptionPreset(
        name="podcast",
        display_name="Podcast",
        description="Long-form clean captions for podcast and interview content.",
        font_name="Montserrat",
        font_size=48,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=3,
        shadow_size=1,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=6,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "energetic": CaptionPreset(
        name="energetic",
        display_name="Energetic",
        description="Gaming/hype green bounce effect.",
        font_name="Montserrat-Black",
        font_size=64,
        primary_color="&H0047FF00",
        secondary_color="&H0047FF00",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=5,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "bounce", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color="&H0047FF00",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "elegant": CaptionPreset(
        name="elegant",
        display_name="Elegant",
        description="Luxury/premium gold serif style.",
        font_name="Playfair Display",
        font_size=48,
        primary_color="&H0037AFD4",
        secondary_color="&H0037AFD4",
        outline_color="&H001A1A1A",
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=2,
        shadow_size=1,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color="&H0037AFD4",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
    "social": CaptionPreset(
        name="social",
        display_name="Social",
        description="Social media karaoke style with purple highlight.",
        font_name="Montserrat-Bold",
        font_size=60,
        primary_color=_WHITE,
        secondary_color="&H00F755A8",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color="&H00F755A8",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="karaoke",
    ),
    "focus": CaptionPreset(
        name="focus",
        display_name="Focus",
        description="One word at a time, big bold impact.",
        font_name="Montserrat-Black",
        font_size=80,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=6,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=3,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
        style="word_by_word",
    ),
    "whisper": CaptionPreset(
        name="whisper",
        display_name="Whisper",
        description="Gentle documentary style with light font.",
        font_name="Montserrat-Light",
        font_size=44,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color="&H66000000",
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=2,
        shadow_size=0,
        alignment=2,
        margin_v=80,
        margin_h=60,
        position="bottom",
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
        style="grouped",
    ),
}

# ---------------------------------------------------------------------------
# Legacy aliases -- map old preset names to their closest new preset.
# ---------------------------------------------------------------------------

LEGACY_ALIASES: dict[str, str] = {
    # Old 18 preset names
    "impact_bold": "contrast",
    "focus_highlight": "hormozi",
    "clean_fade": "clean",
    "bold_box": "boxed",
    "classic_fade": "podcast",
    "cinematic_subtitle": "cinematic",
    "neon_pulse": "neon",
    "minimal_presenter": "whisper",
    "elegant_lower_third": "news",
    "dynamic_karaoke": "karaoke_gold",
    "glow_edge": "whisper",
    "typewriter_clean": "clean",
    "news_strip": "news",
    "retro_arcade": "retro",
    "soft_serif": "elegant",
    "contrast_max": "contrast",
    "split_dual_tone": "social",
    "quiet_doc": "whisper",
    # Common style names
    "mrbeast": "mrbeast",
    "hormozi": "hormozi",
    "tiktok": "boxed",
    "capcut": "clean",
    "netflix": "cinematic",
    "gaming": "energetic",
    "animated": "karaoke_gold",
    "grouped": "clean",
    "word_by_word": "focus",
    "static": "podcast",
    "uppercase": "contrast",
    "lowercase": "clean",
}

CAPTION_STYLE_ALIASES: dict[str, str] = {key: key for key in PRESETS}
CAPTION_STYLE_ALIASES.update(LEGACY_ALIASES)

CAPTION_TEMPLATE_DEFAULTS: CaptionTemplate = {
    "show": False,
    "presetName": "clean",
    "animation": "fade",
    "position": "bottom",
    "fontSize": 48,
    "fontFamily": "Montserrat-Bold",
    "fontColor": _WHITE,
    "highlightColor": _WHITE,
    "strokeColor": _BLACK,
    "shadowColor": _SEMI_BLACK,
    "maxCharsPerCaption": 24,
    "maxLines": 2,
    "lineDelay": 0.05,
    "uppercase": False,
    "wordHighlight": False,
    "backgroundBox": False,
    "style": "grouped",
}

SUPPORTED_CAPTION_STYLES: set[str] = set(PRESETS)


def normalize_caption_style(style: str | None) -> str:
    if not style:
        return CAPTION_TEMPLATE_DEFAULTS["presetName"]
    key = style.strip().lower().replace("-", "_").replace(" ", "_")
    return CAPTION_STYLE_ALIASES.get(key, CAPTION_TEMPLATE_DEFAULTS["presetName"])


def get_preset(name: str) -> CaptionPreset:
    raw_key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    mapped = CAPTION_STYLE_ALIASES.get(raw_key)
    if mapped is None:
        raise KeyError(f"Unknown caption preset: {name!r}")
    preset = PRESETS.get(mapped)
    if preset is None:
        raise KeyError(f"Unknown caption preset: {name!r}")
    return preset


def list_supported_styles() -> list[str]:
    return sorted(SUPPORTED_CAPTION_STYLES)


def list_animation_presets() -> list[str]:
    return list(ANIMATION_OPTIONS)


def _normalize_override_key(key: str) -> str:
    return {
        "displayName": "label",
        "fontName": "font_name",
        "fontSize": "font_size",
        "primaryColor": "primary_color",
        "secondaryColor": "secondary_color",
        "highlightColor": "highlight_color",
        "outlineColor": "outline_color",
        "backColor": "back_color",
        "marginV": "margin_v",
        "marginH": "margin_h",
        "marginL": "margin_l",
        "marginR": "margin_r",
        "wordHighlight": "word_highlight",
        "maxWordsPerLine": "max_words_per_line",
        "maxCharsPerLine": "max_chars_per_line",
        "maxCharsPerCaption": "max_chars_per_line",
        "maxLines": "max_lines",
        "safeMarginX": "safe_margin_x",
        "safeMarginY": "safe_margin_y",
        "safeArea": "safe_area",
        "backgroundBox": "background_box",
        "backgroundPadding": "background_padding",
        "lineDelay": "line_delay",
        "outlineSize": "outline",
        "shadowSize": "shadow",
    }.get(key, key)


def resolve_preset(
    preset_name: str | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_name = normalize_caption_style(preset_name)
    base_preset = PRESETS.get(resolved_name) or PRESETS[CAPTION_TEMPLATE_DEFAULTS["presetName"]]
    base = deepcopy(base_preset.to_style_dict())
    if not overrides:
        return base

    for raw_key, value in overrides.items():
        key_name = _normalize_override_key(str(raw_key))

        if key_name == "animation":
            if isinstance(value, Mapping):
                base["animation"] = _normalize_animation_config(dict(value))
            else:
                current = dict(base.get("animation") or {})
                current["type"] = _normalize_animation(str(value))
                current.setdefault("duration", 0.2)
                current.setdefault("delay_between_words", 0.05)
                base["animation"] = _normalize_animation_config(current)
            continue

        if key_name in {"primary_color", "secondary_color", "outline_color", "back_color", "highlight_color"}:
            fallback = base.get(key_name, "&H00FFFFFF")
            base[key_name] = to_ass_color(str(value), fallback=str(fallback))
            if key_name == "highlight_color" and bool(base.get("word_highlight")):
                base["secondary_color"] = base["highlight_color"]
            continue

        if key_name in {
            "font_size",
            "outline",
            "shadow",
            "margin_v",
            "margin_l",
            "margin_r",
            "margin_h",
            "max_words_per_line",
            "max_chars_per_line",
            "max_lines",
            "safe_margin_x",
            "safe_margin_y",
            "background_padding",
            "alignment",
        }:
            try:
                casted = int(value)
            except (TypeError, ValueError):
                continue
            base[key_name] = casted
            if key_name == "margin_h":
                base["margin_l"] = casted
                base["margin_r"] = casted
            if key_name == "max_words_per_line" and "max_chars_per_line" not in overrides:
                base["max_chars_per_line"] = max(12, casted * 6)
            continue

        if key_name in {
            "bold",
            "italic",
            "word_highlight",
            "background_box",
            "uppercase",
            "safe_area",
        }:
            base[key_name] = bool(value)
            if key_name == "word_highlight" and bool(value):
                base["secondary_color"] = base.get("highlight_color", base["secondary_color"])
            continue

        if key_name in {"line_delay"}:
            base[key_name] = max(0.0, _as_float(value, 0.0))
            continue

        if key_name in {"position", "font_name", "label", "description", "style"}:
            base[key_name] = str(value)
            continue

        base[key_name] = value

    return base


def resolve_caption_preset(preset: str | None) -> dict[str, Any]:
    """Return frontend caption template payload for the chosen preset."""
    cfg = resolve_preset(preset)
    animation = cfg.get("animation") or {"type": "none", "duration": 0.0, "delay_between_words": 0.0}
    return {
        **CAPTION_TEMPLATE_DEFAULTS,
        "show": True,
        "presetName": cfg.get("id", CAPTION_TEMPLATE_DEFAULTS["presetName"]),
        "animation": animation.get("type", "none"),
        "position": cfg.get("position", "auto"),
        "fontSize": int(cfg.get("font_size", CAPTION_TEMPLATE_DEFAULTS["fontSize"])),
        "fontFamily": str(cfg.get("font_name", CAPTION_TEMPLATE_DEFAULTS["fontFamily"])),
        "fontColor": str(cfg.get("primary_color", CAPTION_TEMPLATE_DEFAULTS["fontColor"])),
        "highlightColor": str(cfg.get("highlight_color", CAPTION_TEMPLATE_DEFAULTS["highlightColor"])),
        "strokeColor": str(cfg.get("outline_color", CAPTION_TEMPLATE_DEFAULTS["strokeColor"])),
        "shadowColor": str(cfg.get("back_color", CAPTION_TEMPLATE_DEFAULTS["shadowColor"])),
        "maxCharsPerCaption": int(
            cfg.get("max_chars_per_line", CAPTION_TEMPLATE_DEFAULTS["maxCharsPerCaption"])
        ),
        "maxLines": int(cfg.get("max_lines", CAPTION_TEMPLATE_DEFAULTS["maxLines"])),
        "lineDelay": float(
            animation.get("delay_between_words", cfg.get("line_delay", CAPTION_TEMPLATE_DEFAULTS["lineDelay"]))
        ),
        "uppercase": bool(cfg.get("uppercase", CAPTION_TEMPLATE_DEFAULTS["uppercase"])),
        "wordHighlight": bool(cfg.get("word_highlight", CAPTION_TEMPLATE_DEFAULTS["wordHighlight"])),
        "backgroundBox": bool(cfg.get("background_box", CAPTION_TEMPLATE_DEFAULTS["backgroundBox"])),
        "style": cfg.get("style", "grouped"),
    }


def list_caption_presets() -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = []
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        style = preset.to_style_dict()
        presets.append(
            {
                "id": preset.name,
                "label": preset.display_name,
                "description": preset.description,
                "captions": resolve_caption_preset(preset.name),
                "style": {
                    "font_name": style.get("font_name"),
                    "font_size": style.get("font_size"),
                    "primary_color": style.get("primary_color"),
                    "secondary_color": style.get("secondary_color"),
                    "outline_color": style.get("outline_color"),
                    "back_color": style.get("back_color"),
                    "alignment": style.get("alignment"),
                    "margin_v": style.get("margin_v"),
                    "margin_h": preset.margin_h,
                    "max_words_per_line": style.get("max_words_per_line"),
                    "animation": style.get("animation"),
                    "word_highlight": style.get("word_highlight"),
                    "highlight_color": style.get("highlight_color"),
                    "background_box": style.get("background_box"),
                    "background_padding": style.get("background_padding"),
                    "uppercase": style.get("uppercase"),
                    "safe_area": style.get("safe_area"),
                    "style": style.get("style"),
                },
            }
        )
    return presets


def _hex_to_ass(hex_color: str) -> str:
    raw = hex_color.strip().lstrip("#")
    if len(raw) == 6:
        r, g, b = raw[0:2], raw[2:4], raw[4:6]
        return f"&H00{b}{g}{r}".upper()
    if len(raw) == 8:
        a, r, g, b = raw[0:2], raw[2:4], raw[4:6], raw[6:8]
        return f"&H{a}{b}{g}{r}".upper()
    return "&H00FFFFFF"


def to_ass_color(value: str | None, fallback: str = "&H00FFFFFF") -> str:
    """Best-effort conversion from named/css/ASS colors to ASS format."""
    if not value:
        return fallback
    token = value.strip()
    if token.startswith("&H"):
        return token.upper().rstrip("&")
    if token.startswith("#"):
        return _hex_to_ass(token)
    named: dict[str, str] = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "yellow": "&H0000FFFF",
        "cyan": "&H00FFFF00",
        "magenta": "&H00FF00FF",
        "orange": "&H0000A5FF",
        "gray": "&H00808080",
        "grey": "&H00808080",
    }
    return named.get(token.lower(), fallback)


# Backward-compatible mapping used across existing workers.
CAPTION_PRESETS: dict[str, dict[str, Any]] = {
    key: value.to_style_dict() for key, value in PRESETS.items()
}


__all__ = [
    "ANIMATION_ALIASES",
    "ANIMATION_OPTIONS",
    "AnimationType",
    "CAPTION_PRESETS",
    "CAPTION_STYLE_ALIASES",
    "CAPTION_TEMPLATE_DEFAULTS",
    "LEGACY_ALIASES",
    "SUPPORTED_CAPTION_STYLES",
    "CaptionAnimation",
    "CaptionPreset",
    "CaptionTemplate",
    "PRESETS",
    "get_preset",
    "list_animation_presets",
    "list_caption_presets",
    "list_supported_styles",
    "normalize_caption_style",
    "resolve_caption_preset",
    "resolve_preset",
    "to_ass_color",
]

"""Caption preset catalog and normalization helpers for ASS rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, TypedDict

AnimationType = Literal["fade", "slide_up", "pop", "karaoke", "scale", "none"]


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
        }


ANIMATION_OPTIONS: tuple[AnimationType, ...] = (
    "none",
    "fade",
    "slide_up",
    "pop",
    "karaoke",
    "scale",
)

ANIMATION_ALIASES: dict[str, AnimationType] = {
    "none": "none",
    "off": "none",
    "no_animation": "none",
    "fade": "fade",
    "fade_in": "fade",
    "slide": "slide_up",
    "slide_up": "slide_up",
    "slideup": "slide_up",
    "pop": "pop",
    "punch": "pop",
    "bounce": "pop",
    "karaoke": "karaoke",
    "word_highlight": "karaoke",
    "scale": "scale",
    "zoom": "scale",
    "zoom_in": "scale",
}

_WHITE = "&H00FFFFFF"
_BLACK = "&H00000000"
_SEMI_BLACK = "&H80000000"
_YELLOW = "&H0000FFFF"
_CYAN = "&H00FFFF00"


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


PRESETS: dict[str, CaptionPreset] = {
    "impact_bold": CaptionPreset(
        name="impact_bold",
        display_name="Impact Bold",
        description="Large all-caps impact text with thick outline and pop motion.",
        font_name="Montserrat-ExtraBold",
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
        max_words_per_line=4,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "focus_highlight": CaptionPreset(
        name="focus_highlight",
        display_name="Focus Highlight",
        description="Bold high-contrast captions with per-word yellow emphasis.",
        font_name="Montserrat-Bold",
        font_size=74,
        primary_color=_WHITE,
        secondary_color=_YELLOW,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=1,
        alignment=2,
        margin_v=88,
        margin_h=60,
        max_words_per_line=4,
        animation={"type": "karaoke", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "clean_fade": CaptionPreset(
        name="clean_fade",
        display_name="Clean Fade",
        description="Minimal clean subtitle look with soft fade transition.",
        font_name="Montserrat-SemiBold",
        font_size=64,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=2,
        shadow_size=0,
        alignment=2,
        margin_v=84,
        margin_h=56,
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "bold_box": CaptionPreset(
        name="bold_box",
        display_name="Bold Box",
        description="Bold captions with dark background slab and slide-up motion.",
        font_name="Montserrat-ExtraBold",
        font_size=72,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=3,
        shadow_size=1,
        alignment=2,
        margin_v=92,
        margin_h=56,
        max_words_per_line=4,
        animation={"type": "slide_up", "duration": 0.22, "delay_between_words": 0.05},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=True,
        background_padding=15,
        uppercase=True,
        safe_area=True,
    ),
    "classic_fade": CaptionPreset(
        name="classic_fade",
        display_name="Classic Fade",
        description="Balanced medium bold style with subtle outline and fade.",
        font_name="Montserrat-Bold",
        font_size=60,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=3,
        shadow_size=0,
        alignment=2,
        margin_v=86,
        margin_h=56,
        max_words_per_line=4,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "cinematic_subtitle": CaptionPreset(
        name="cinematic_subtitle",
        display_name="Cinematic Subtitle",
        description="Smaller cinematic subtitle style with minimal treatment.",
        font_name="Montserrat-SemiBold",
        font_size=48,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=2,
        shadow_size=0,
        alignment=2,
        margin_v=72,
        margin_h=56,
        max_words_per_line=6,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "neon_pulse": CaptionPreset(
        name="neon_pulse",
        display_name="Neon Pulse",
        description="High-energy captions with bright cyan emphasis and punchy pop.",
        font_name="Montserrat-ExtraBold",
        font_size=74,
        primary_color=_WHITE,
        secondary_color=_CYAN,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=5,
        shadow_size=1,
        alignment=2,
        margin_v=88,
        margin_h=58,
        max_words_per_line=4,
        animation={"type": "pop", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color=_CYAN,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "minimal_presenter": CaptionPreset(
        name="minimal_presenter",
        display_name="Minimal Presenter",
        description="Ultra-clean presentation style with no outline or shadow.",
        font_name="Montserrat-Medium",
        font_size=56,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=0,
        shadow_size=0,
        alignment=2,
        margin_v=84,
        margin_h=60,
        max_words_per_line=5,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.03},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "elegant_lower_third": CaptionPreset(
        name="elegant_lower_third",
        display_name="Elegant Lower Third",
        description="Elegant lower-third style with extra bottom spacing and box.",
        font_name="Montserrat-SemiBold",
        font_size=58,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=2,
        shadow_size=0,
        alignment=2,
        margin_v=140,
        margin_h=70,
        max_words_per_line=5,
        animation={"type": "fade", "duration": 0.2, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=True,
        background_padding=14,
        uppercase=False,
        safe_area=True,
    ),
    "dynamic_karaoke": CaptionPreset(
        name="dynamic_karaoke",
        display_name="Dynamic Karaoke",
        description="Karaoke highlight effect tuned for viral short-form pacing.",
        font_name="Montserrat-ExtraBold",
        font_size=72,
        primary_color=_WHITE,
        secondary_color=_YELLOW,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=1,
        alignment=2,
        margin_v=90,
        margin_h=58,
        max_words_per_line=4,
        animation={"type": "karaoke", "duration": 0.2, "delay_between_words": 0.08},
        word_highlight=True,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "glow_edge": CaptionPreset(
        name="glow_edge",
        display_name="Glow Edge",
        description="Soft glow-like edge treatment with smooth fade timing.",
        font_name="Montserrat-SemiBold",
        font_size=62,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color="&H00303030",
        back_color="&H70303030",
        bold=True,
        italic=False,
        outline_size=1,
        shadow_size=2,
        alignment=2,
        margin_v=84,
        margin_h=58,
        max_words_per_line=5,
        animation={"type": "fade", "duration": 0.24, "delay_between_words": 0.04},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "typewriter_clean": CaptionPreset(
        name="typewriter_clean",
        display_name="Typewriter Clean",
        description="Word-by-word reveal style while preserving a clean mono color look.",
        font_name="Montserrat-Medium",
        font_size=58,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=False,
        italic=False,
        outline_size=1,
        shadow_size=0,
        alignment=2,
        margin_v=86,
        margin_h=58,
        max_words_per_line=5,
        animation={"type": "karaoke", "duration": 0.2, "delay_between_words": 0.06},
        word_highlight=True,
        highlight_color=_WHITE,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "news_strip": CaptionPreset(
        name="news_strip",
        display_name="News Strip",
        description="Lower strip caption block optimized for dense informational speech.",
        font_name="Montserrat-SemiBold",
        font_size=52,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color="&HC0202020",
        bold=True,
        italic=False,
        outline_size=1,
        shadow_size=0,
        alignment=2,
        margin_v=56,
        margin_h=76,
        max_words_per_line=6,
        animation={"type": "slide_up", "duration": 0.18, "delay_between_words": 0.03},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=True,
        background_padding=18,
        uppercase=False,
        safe_area=True,
    ),
    "retro_arcade": CaptionPreset(
        name="retro_arcade",
        display_name="Retro Arcade",
        description="Retro neon arcade flavor with punchy scale pop timing.",
        font_name="Courier New Bold",
        font_size=64,
        primary_color="&H00A8FF7A",
        secondary_color=_CYAN,
        outline_color="&H00601020",
        back_color="&H80101010",
        bold=True,
        italic=False,
        outline_size=4,
        shadow_size=2,
        alignment=2,
        margin_v=88,
        margin_h=62,
        max_words_per_line=4,
        animation={"type": "scale", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color=_CYAN,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "soft_serif": CaptionPreset(
        name="soft_serif",
        display_name="Soft Serif",
        description="Elegant serif subtitle style for narrative and interview formats.",
        font_name="Georgia",
        font_size=50,
        primary_color="&H00F8F8F2",
        secondary_color="&H00F8F8F2",
        outline_color="&H00101010",
        back_color="&H70000000",
        bold=False,
        italic=False,
        outline_size=1,
        shadow_size=0,
        alignment=2,
        margin_v=82,
        margin_h=64,
        max_words_per_line=6,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "contrast_max": CaptionPreset(
        name="contrast_max",
        display_name="Contrast Max",
        description="Accessibility-first style with high contrast and heavy outline.",
        font_name="Montserrat-Bold",
        font_size=68,
        primary_color=_WHITE,
        secondary_color=_WHITE,
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=7,
        shadow_size=0,
        alignment=2,
        margin_v=86,
        margin_h=58,
        max_words_per_line=4,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=True,
        safe_area=True,
    ),
    "split_dual_tone": CaptionPreset(
        name="split_dual_tone",
        display_name="Split Dual Tone",
        description="Dual-tone highlight rhythm with karaoke timing for modern edits.",
        font_name="Montserrat-Bold",
        font_size=66,
        primary_color=_WHITE,
        secondary_color="&H00A5FF00",
        outline_color=_BLACK,
        back_color=_SEMI_BLACK,
        bold=True,
        italic=False,
        outline_size=3,
        shadow_size=1,
        alignment=2,
        margin_v=86,
        margin_h=60,
        max_words_per_line=4,
        animation={"type": "karaoke", "duration": 0.2, "delay_between_words": 0.05},
        word_highlight=True,
        highlight_color="&H00A5FF00",
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
    "quiet_doc": CaptionPreset(
        name="quiet_doc",
        display_name="Quiet Doc",
        description="Low-noise documentary captions designed for long-form readability.",
        font_name="Montserrat-Regular",
        font_size=44,
        primary_color="&H00EDEDED",
        secondary_color="&H00EDEDED",
        outline_color="&H000F0F0F",
        back_color="&H60000000",
        bold=False,
        italic=False,
        outline_size=1,
        shadow_size=0,
        alignment=2,
        margin_v=72,
        margin_h=62,
        max_words_per_line=7,
        animation={"type": "none", "duration": 0.0, "delay_between_words": 0.0},
        word_highlight=False,
        highlight_color=_YELLOW,
        background_box=False,
        background_padding=0,
        uppercase=False,
        safe_area=True,
    ),
}

CAPTION_STYLE_ALIASES: dict[str, str] = {key: key for key in PRESETS}
CAPTION_STYLE_ALIASES.update(
    {
        "impact": "impact_bold",
        "focus": "focus_highlight",
        "clean": "clean_fade",
        "box": "bold_box",
        "classic": "classic_fade",
        "cinematic": "cinematic_subtitle",
        "neon": "neon_pulse",
        "presenter": "minimal_presenter",
        "lower_third": "elegant_lower_third",
        "karaoke": "dynamic_karaoke",
        "glow": "glow_edge",
        "typewriter": "typewriter_clean",
        "news": "news_strip",
        "retro": "retro_arcade",
        "serif": "soft_serif",
        "accessibility": "contrast_max",
        "high_contrast": "contrast_max",
        "dual_tone": "split_dual_tone",
        "quiet": "quiet_doc",
        "documentary_quiet": "quiet_doc",
        # Backward compatibility for older IDs and styles.
        "mrbeast": "impact_bold",
        "mr_beast": "impact_bold",
        "mrbeast_style": "impact_bold",
        "hormozi": "focus_highlight",
        "alex_hormozi": "focus_highlight",
        "hormozi_style": "focus_highlight",
        "iman_gadzhi": "clean_fade",
        "iman": "clean_fade",
        "iman_style": "clean_fade",
        "tiktok_bold": "bold_box",
        "tiktok": "bold_box",
        "modern_tiktok": "bold_box",
        "capcut_default": "classic_fade",
        "capcut": "classic_fade",
        "netflix_documentary": "cinematic_subtitle",
        "netflix": "cinematic_subtitle",
        "gaming_streamer": "neon_pulse",
        "gaming": "neon_pulse",
        "apple_keynote": "minimal_presenter",
        "apple": "minimal_presenter",
        "cinematic_lower_third": "elegant_lower_third",
        "viral_shorts_highlight": "dynamic_karaoke",
        "viral": "dynamic_karaoke",
        "shorts_highlight": "dynamic_karaoke",
        "clean_minimal": "clean_fade",
        "bold_tiktok": "bold_box",
        "word_highlighted": "dynamic_karaoke",
        "block_background": "bold_box",
        "animated": "dynamic_karaoke",
        "grouped": "classic_fade",
        "word_by_word": "dynamic_karaoke",
        "progressive": "impact_bold",
        "punctuated": "classic_fade",
        "static": "classic_fade",
        "two_line": "elegant_lower_third",
        "uppercase": "impact_bold",
        "lowercase": "clean_fade",
        "headline": "elegant_lower_third",
        "minimal": "clean_fade",
        "focus_word": "focus_highlight",
        "build": "impact_bold",
        "split": "elegant_lower_third",
        "sentence": "classic_fade",
        "caps": "impact_bold",
        "lower": "clean_fade",
        "title_case": "minimal_presenter",
    }
)

CAPTION_TEMPLATE_DEFAULTS: CaptionTemplate = {
    "show": False,
    "presetName": "classic_fade",
    "animation": "fade",
    "position": "bottom",
    "fontSize": 60,
    "fontFamily": "Montserrat-Bold",
    "fontColor": _WHITE,
    "highlightColor": _YELLOW,
    "strokeColor": _BLACK,
    "shadowColor": _SEMI_BLACK,
    "maxCharsPerCaption": 24,
    "maxLines": 2,
    "lineDelay": 0.05,
    "uppercase": False,
    "wordHighlight": False,
    "backgroundBox": False,
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

        if key_name in {"position", "font_name", "label", "description"}:
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

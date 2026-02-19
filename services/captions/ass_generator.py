"""ASS subtitle generation from transcript JSON and style presets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from services.captions.caption_presets import CaptionPreset, resolve_preset

_MIN_EVENT_DURATION_SECONDS = 0.06
_DEFAULT_PLAY_RES = (1080, 1920)
_PLAY_RES_BY_ASPECT: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}
_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class WordToken:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentToken:
    start: float
    end: float
    text: str
    words: list[WordToken]


@dataclass(frozen=True)
class DialogueEvent:
    start: float
    end: float
    text: str
    animation_tag: str


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(text: str, cleanup_punctuation: bool) -> str:
    cleaned = _WS_RE.sub(" ", text or "").strip()
    if cleanup_punctuation:
        cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


def _escape_ass_text(text: str) -> str:
    escaped = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return escaped.replace("\n", r"\N")


def format_ass_timestamp(seconds: float) -> str:
    safe = max(0.0, float(seconds))
    total_cs = int(round(safe * 100.0))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centis = total_cs % 100
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _normalize_transcript(transcript_json: dict[str, Any] | list[dict[str, Any]]) -> list[SegmentToken]:
    if isinstance(transcript_json, list):
        source_segments = transcript_json
    else:
        source_segments = transcript_json.get("segments") or []

    normalized: list[SegmentToken] = []
    for raw in source_segments:
        start = _to_float(raw.get("start"))
        end = _to_float(raw.get("end"))
        if end <= start:
            continue

        raw_words = raw.get("words") or []
        words: list[WordToken] = []
        if isinstance(raw_words, list):
            for item in raw_words:
                word_text = str(item.get("word", item.get("text", ""))).strip()
                if not word_text:
                    continue
                w_start = _to_float(item.get("start"), start)
                w_end = _to_float(item.get("end"), end)
                if w_end <= w_start:
                    continue
                words.append(WordToken(word=word_text, start=max(start, w_start), end=min(end, w_end)))
            words.sort(key=lambda w: (w.start, w.end))

        text = str(raw.get("text") or "").strip()
        if not text and words:
            text = " ".join(word.word for word in words)
        if not text:
            continue

        normalized.append(
            SegmentToken(
                start=start,
                end=end,
                text=text,
                words=words,
            )
        )

    normalized.sort(key=lambda segment: (segment.start, segment.end))
    return normalized


def _words_for_line_fallback(segment: SegmentToken) -> list[WordToken]:
    if segment.words:
        return segment.words

    tokens = [token for token in segment.text.split(" ") if token.strip()]
    if not tokens:
        return []

    duration = max(_MIN_EVENT_DURATION_SECONDS, segment.end - segment.start)
    per_word = duration / max(1, len(tokens))
    words: list[WordToken] = []
    for index, token in enumerate(tokens):
        start = segment.start + (index * per_word)
        end = segment.start + ((index + 1) * per_word)
        words.append(WordToken(word=token, start=start, end=min(segment.end, end)))
    return words


def _chunk_tokens(
    words: list[WordToken],
    *,
    max_chars_per_line: int,
    max_lines: int,
) -> list[list[list[WordToken]]]:
    pages: list[list[list[WordToken]]] = []
    if not words:
        return pages

    safe_max_chars = max(8, max_chars_per_line)
    safe_max_lines = max(1, max_lines)

    current_page: list[list[WordToken]] = [[]]
    current_line_chars = 0

    def flush_page() -> None:
        nonlocal current_page, current_line_chars
        trimmed = [line for line in current_page if line]
        if trimmed:
            pages.append(trimmed)
        current_page = [[]]
        current_line_chars = 0

    for word in words:
        token_len = len(word.word)
        needs_space = 1 if current_page[-1] else 0
        projected = current_line_chars + needs_space + token_len

        if projected <= safe_max_chars or not current_page[-1]:
            current_page[-1].append(word)
            current_line_chars = projected
            continue

        if len(current_page) < safe_max_lines:
            current_page.append([word])
            current_line_chars = token_len
            continue

        flush_page()
        current_page[-1].append(word)
        current_line_chars = token_len

    flush_page()
    return pages


def _line_text(words: Iterable[WordToken], *, uppercase: bool, cleanup_punctuation: bool) -> str:
    text = " ".join(word.word for word in words)
    text = _clean_text(text, cleanup_punctuation)
    if uppercase:
        text = text.upper()
    return text


def _page_start_end(page: list[list[WordToken]], segment: SegmentToken) -> tuple[float, float]:
    flat = [word for line in page for word in line]
    if not flat:
        return segment.start, segment.end
    start = min(word.start for word in flat)
    end = max(word.end for word in flat)
    return start, max(start + _MIN_EVENT_DURATION_SECONDS, end)


def _alignment_column(alignment: int) -> int:
    if alignment in {1, 4, 7}:
        return 1
    if alignment in {3, 6, 9}:
        return 3
    return 2


def _alignment_row(alignment: int) -> int:
    if alignment in {7, 8, 9}:
        return 3
    if alignment in {4, 5, 6}:
        return 2
    return 1


def _compose_alignment(column: int, row: int) -> int:
    row = min(3, max(1, row))
    column = min(3, max(1, column))
    if row == 1:
        return {1: 1, 2: 2, 3: 3}[column]
    if row == 2:
        return {1: 4, 2: 5, 3: 6}[column]
    return {1: 7, 2: 8, 3: 9}[column]


def _resolve_alignment_and_margin(
    preset: CaptionPreset,
    play_res: tuple[int, int],
) -> tuple[int, int]:
    alignment = _to_int(preset.get("alignment"), 2)
    margin_v = _to_int(preset.get("margin_v"), 80)
    safe_margin_y = _to_int(preset.get("safe_margin_y"), margin_v)

    position = str(preset.get("position") or "auto").lower().strip()
    if position == "auto":
        return alignment, max(0, margin_v)

    column = _alignment_column(alignment)
    if position == "top":
        return _compose_alignment(column, 3), max(0, safe_margin_y)
    if position == "middle":
        return _compose_alignment(column, 2), max(0, play_res[1] // 3)
    return _compose_alignment(column, 1), max(0, margin_v)


def _animation_tag(
    preset: CaptionPreset,
    *,
    play_res: tuple[int, int],
) -> str:
    animation = preset.get("animation") or {"type": "none", "duration": 0.0}
    animation_type = str(animation.get("type", "none"))
    duration_ms = max(0, int(round(_to_float(animation.get("duration"), 0.0) * 1000.0)))
    if animation_type == "fade":
        fade_ms = max(1, duration_ms or 200)
        return rf"{{\fad({fade_ms},{fade_ms})}}"

    if animation_type == "slide_up":
        alignment = _to_int(preset.get("alignment"), 2)
        column = _alignment_column(alignment)
        margin_l = _to_int(preset.get("margin_l"), _to_int(preset.get("safe_margin_x"), 56))
        margin_r = _to_int(preset.get("margin_r"), _to_int(preset.get("safe_margin_x"), 56))
        margin_v = _to_int(preset.get("margin_v"), 80)

        if column == 1:
            x = margin_l + 200
        elif column == 3:
            x = play_res[0] - margin_r - 200
        else:
            x = play_res[0] // 2

        end_y = play_res[1] - margin_v
        start_y = end_y + int(play_res[1] * 0.09)
        return rf"{{\move({x},{start_y},{x},{end_y})}}"

    if animation_type == "pop":
        pop_ms = max(1, duration_ms or 200)
        return rf"{{\t(0,{pop_ms},\fscx120\fscy120)}}"

    return ""


def _karaoke_text(
    page: list[list[WordToken]],
    *,
    uppercase: bool,
    cleanup_punctuation: bool,
) -> str:
    lines: list[str] = []
    for line_words in page:
        fragments: list[str] = []
        for word in line_words:
            token = _line_text([word], uppercase=uppercase, cleanup_punctuation=cleanup_punctuation)
            centis = max(1, int(round((word.end - word.start) * 100.0)))
            fragments.append(rf"{{\k{centis}}}{_escape_ass_text(token)}")
        lines.append(" ".join(fragment for fragment in fragments if fragment))
    return r"\N".join(lines)


def _static_text(
    page: list[list[WordToken]],
    *,
    uppercase: bool,
    cleanup_punctuation: bool,
) -> str:
    lines = [
        _escape_ass_text(
            _line_text(line_words, uppercase=uppercase, cleanup_punctuation=cleanup_punctuation)
        )
        for line_words in page
    ]
    lines = [line for line in lines if line]
    return r"\N".join(lines)


def _build_events(
    transcript_segments: list[SegmentToken],
    preset: CaptionPreset,
    *,
    play_res: tuple[int, int],
) -> list[DialogueEvent]:
    max_chars = _to_int(preset.get("max_chars_per_line"), 30)
    max_lines = _to_int(preset.get("max_lines"), 2)
    uppercase = bool(preset.get("uppercase", False))
    cleanup_punctuation = bool(preset.get("punctuation_cleanup", True))
    line_delay = max(0.0, _to_float(preset.get("line_delay"), 0.0))
    wants_word_highlight = bool(preset.get("word_highlight", False))
    animation_tag = _animation_tag(preset, play_res=play_res)

    events: list[DialogueEvent] = []
    event_index = 0
    for segment in transcript_segments:
        words = _words_for_line_fallback(segment)
        if not words:
            continue
        pages = _chunk_tokens(words, max_chars_per_line=max_chars, max_lines=max_lines)
        if not pages:
            continue

        use_karaoke = wants_word_highlight and bool(segment.words)
        for page in pages:
            start, end = _page_start_end(page, segment)
            delay_seconds = event_index * line_delay
            start += delay_seconds
            end += delay_seconds
            if end <= start:
                end = start + _MIN_EVENT_DURATION_SECONDS

            text = (
                _karaoke_text(page, uppercase=uppercase, cleanup_punctuation=cleanup_punctuation)
                if use_karaoke
                else _static_text(page, uppercase=uppercase, cleanup_punctuation=cleanup_punctuation)
            )
            if not text:
                continue
            events.append(
                DialogueEvent(
                    start=start,
                    end=end,
                    text=text,
                    animation_tag=animation_tag,
                )
            )
            event_index += 1
    return events


def _style_line(
    preset: CaptionPreset,
    *,
    play_res: tuple[int, int],
) -> str:
    alignment, margin_v = _resolve_alignment_and_margin(preset, play_res)
    margin_l = _to_int(preset.get("margin_l"), _to_int(preset.get("safe_margin_x"), 56))
    margin_r = _to_int(preset.get("margin_r"), _to_int(preset.get("safe_margin_x"), 56))
    outline = max(0, _to_int(preset.get("outline"), 3))
    shadow = max(0, _to_int(preset.get("shadow"), 1))
    border_style = 3 if bool(preset.get("background_box", False)) else 1
    bold_flag = -1 if bool(preset.get("bold", True)) else 0
    italic_flag = -1 if bool(preset.get("italic", False)) else 0
    underline_flag = -1 if bool(preset.get("underline", False)) else 0

    font_size = _to_int(preset.get("font_size"), 64)
    if play_res[0] >= 1920 and play_res[1] <= 1080:
        font_size = max(36, int(math.floor(font_size * 0.72)))

    return (
        "Style: Default,"
        f"{preset.get('font_name', 'Montserrat-Bold')},{font_size},"
        f"{preset.get('primary_color', '&H00FFFFFF')},"
        f"{preset.get('secondary_color', '&H0000C8FF')},"
        f"{preset.get('outline_color', '&H00000000')},"
        f"{preset.get('back_color', '&H80000000')},"
        f"{bold_flag},{italic_flag},{underline_flag},0,100,100,0,0,"
        f"{border_style},{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1"
    )


def _play_res(video_aspect_ratio: str) -> tuple[int, int]:
    return _PLAY_RES_BY_ASPECT.get(video_aspect_ratio, _DEFAULT_PLAY_RES)


def generate_ass_content(
    transcript_json: dict[str, Any] | list[dict[str, Any]],
    preset_name: str,
    *,
    video_aspect_ratio: str = "9:16",
    overrides: dict[str, Any] | None = None,
) -> str:
    """Build ASS file content from transcript JSON and preset id."""
    preset = resolve_preset(preset_name, overrides=overrides)
    play_res = _play_res(video_aspect_ratio)
    segments = _normalize_transcript(transcript_json)
    events = _build_events(segments, preset, play_res=play_res)

    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res[0]}",
        f"PlayResY: {play_res[1]}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(preset, play_res=play_res),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for event in events:
        start = format_ass_timestamp(event.start)
        end = format_ass_timestamp(event.end)
        text = f"{event.animation_tag}{event.text}" if event.animation_tag else event.text
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def generate_ass_file(
    *,
    transcript_json: dict[str, Any] | list[dict[str, Any]],
    preset_name: str,
    output_path: str,
    video_aspect_ratio: str = "9:16",
    overrides: dict[str, Any] | None = None,
) -> str:
    """Write generated ASS content to disk and return the output path."""
    content = generate_ass_content(
        transcript_json,
        preset_name,
        video_aspect_ratio=video_aspect_ratio,
        overrides=overrides,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


__all__ = [
    "generate_ass_content",
    "generate_ass_file",
    "format_ass_timestamp",
]

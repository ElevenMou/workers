"""Caption line builder functions for ASS rendering."""

import math

from services.captions.effects import _apply_animation
from services.captions.segments import _words_from_segment


def _fmt_ts(seconds: float) -> str:
    """Format seconds to ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass_text(text: str) -> str:
    """Escape text so it is safe inside ASS dialogue payload."""
    return text.replace("\\", "\\\\").replace("{", "").replace("}", "")


def _text_transform(text: str, mode: str) -> str:
    if mode == "uppercase":
        return text.upper()
    if mode == "lowercase":
        return text.lower()
    if mode == "headline":
        return text.title()
    return text


def _split_words_evenly(words: list[str], lines_per_page: int) -> list[str]:
    if not words:
        return []
    if lines_per_page <= 1:
        return [" ".join(words).strip()]
    per_line = max(1, math.ceil(len(words) / lines_per_page))
    return [
        " ".join(words[i : i + per_line]).strip()
        for i in range(0, len(words), per_line)
    ]


def _format_caption_lines(lines: list[str], font_case: str) -> str:
    formatted = []
    for line in lines:
        text = _text_transform(line.strip(), font_case)
        text = _escape_ass_text(text)
        if text:
            formatted.append(text)
    return r"\N".join(formatted)


def build_animated_lines(
    segments: list[dict],
    max_words_per_line: int,
    *,
    font_case: str = "as_typed",
    animation: str = "none",
) -> list[str]:
    """Build ASS Dialogue lines with ``\\k`` karaoke tags."""
    lines: list[str] = []

    for seg in segments:
        words = _words_from_segment(seg)
        if not words:
            continue

        for g_start in range(0, len(words), max_words_per_line):
            group = words[g_start : g_start + max_words_per_line]
            if not group:
                continue

            line_start = group[0]["start"]
            line_end = group[-1]["end"]

            parts: list[str] = []
            for w in group:
                duration_cs = max(1, int((w["end"] - w["start"]) * 100))
                word_text = _text_transform(w["word"], font_case)
                word_text = _escape_ass_text(word_text)
                parts.append(f"{{\\kf{duration_cs}}}{word_text}")

            text = " ".join(parts)
            text = _apply_animation(text, animation, line_start, line_end)
            lines.append(
                f"Dialogue: 0,{_fmt_ts(line_start)},{_fmt_ts(line_end)},"
                f"Default,,0,0,0,,{text}"
            )

    return lines


def build_grouped_lines(
    segments: list[dict],
    max_words_per_line: int,
    transform: str | None = None,
    *,
    lines_per_page: int = 1,
    animation: str = "none",
) -> list[str]:
    """Build plain ASS Dialogue lines grouped by number of words."""
    lines: list[str] = []

    for seg in segments:
        words = _words_from_segment(seg)
        if not words:
            continue

        for g_start in range(0, len(words), max_words_per_line):
            group = words[g_start : g_start + max_words_per_line]
            if not group:
                continue

            line_start = group[0]["start"]
            line_end = group[-1]["end"]
            raw_words = [w["word"] for w in group]
            lines_for_page = _split_words_evenly(raw_words, lines_per_page)
            text = _format_caption_lines(lines_for_page, transform or "as_typed")
            if not text:
                continue
            text = _apply_animation(text, animation, line_start, line_end)

            lines.append(
                f"Dialogue: 0,{_fmt_ts(line_start)},{_fmt_ts(line_end)},"
                f"Default,,0,0,0,,{text}"
            )

    return lines


def build_word_lines(
    segments: list[dict],
    *,
    font_case: str = "as_typed",
    animation: str = "none",
) -> list[str]:
    """Build ASS Dialogue lines with one word per subtitle event."""
    lines: list[str] = []

    for seg in segments:
        for word in _words_from_segment(seg):
            text = _text_transform(word["word"], font_case)
            text = _escape_ass_text(text)
            if not text:
                continue
            start = float(word["start"])
            end = max(start + 0.01, float(word["end"]))
            text = _apply_animation(text, animation, start, end)
            lines.append(
                f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},"
                f"Default,,0,0,0,,{text}"
            )

    return lines


def build_progressive_lines(
    segments: list[dict],
    max_words_per_line: int,
    *,
    font_case: str = "as_typed",
    animation: str = "none",
) -> list[str]:
    """Build lines where each event reveals one more word."""
    lines: list[str] = []

    for seg in segments:
        words = _words_from_segment(seg)
        if not words:
            continue

        for g_start in range(0, len(words), max_words_per_line):
            group = words[g_start : g_start + max_words_per_line]
            if not group:
                continue

            for i, word in enumerate(group):
                start = float(word["start"])
                if i + 1 < len(group):
                    end = float(group[i + 1]["start"])
                else:
                    end = float(word["end"])
                end = max(start + 0.01, end)
                phrase = " ".join(w["word"] for w in group[: i + 1]).strip()
                phrase = _text_transform(phrase, font_case)
                phrase = _escape_ass_text(phrase)
                if not phrase:
                    continue
                phrase = _apply_animation(phrase, animation, start, end)
                lines.append(
                    f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},"
                    f"Default,,0,0,0,,{phrase}"
                )

    return lines


def build_two_line_grouped_lines(
    segments: list[dict],
    max_words_per_line: int,
    *,
    transform: str = "as_typed",
    animation: str = "none",
) -> list[str]:
    """Build grouped lines split into two balanced lines."""
    lines: list[str] = []

    for seg in segments:
        words = _words_from_segment(seg)
        if not words:
            continue

        for g_start in range(0, len(words), max_words_per_line):
            group = words[g_start : g_start + max_words_per_line]
            if not group:
                continue

            line_start = group[0]["start"]
            line_end = group[-1]["end"]
            mid = max(1, len(group) // 2)
            top = " ".join(w["word"] for w in group[:mid]).strip()
            bottom = " ".join(w["word"] for w in group[mid:]).strip()
            text = _format_caption_lines([top, bottom], transform)
            if not text:
                continue
            text = _apply_animation(text, animation, line_start, line_end)

            lines.append(
                f"Dialogue: 0,{_fmt_ts(line_start)},{_fmt_ts(line_end)},"
                f"Default,,0,0,0,,{text}"
            )

    return lines


def build_punctuated_lines(
    segments: list[dict],
    *,
    font_case: str = "as_typed",
    animation: str = "none",
) -> list[str]:
    """Split each segment by punctuation and allocate proportional timing."""
    lines: list[str] = []
    punct = {".", ",", "!", "?", ";", ":"}

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        tokens: list[str] = []
        current = []
        for ch in text:
            current.append(ch)
            if ch in punct:
                chunk = "".join(current).strip()
                if chunk:
                    tokens.append(chunk)
                current = []
        tail = "".join(current).strip()
        if tail:
            tokens.append(tail)
        if not tokens:
            tokens = [text]

        start = float(seg["start"])
        end = float(seg["end"])
        duration = max(0.01, end - start)
        total_chars = sum(len(t) for t in tokens) or 1

        cursor = start
        for i, token in enumerate(tokens):
            frac = len(token) / total_chars
            token_dur = duration * frac
            token_start = cursor
            token_end = end if i == len(tokens) - 1 else cursor + token_dur
            cursor = token_end
            token = _text_transform(token, font_case)
            t = _escape_ass_text(token)
            if not t:
                continue
            t = _apply_animation(t, animation, token_start, token_end)
            lines.append(
                f"Dialogue: 0,{_fmt_ts(token_start)},{_fmt_ts(token_end)},"
                f"Default,,0,0,0,,{t}"
            )

    return lines


def build_static_lines(
    segments: list[dict],
    transform: str | None = None,
    *,
    lines_per_page: int = 1,
    animation: str = "none",
) -> list[str]:
    """Build plain ASS Dialogue lines, one per transcript segment."""
    lines: list[str] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        words = text.split()
        line_parts = _split_words_evenly(words, lines_per_page)
        text = _format_caption_lines(line_parts, transform or "as_typed")
        if not text:
            continue
        text = _apply_animation(text, animation, start, end)
        lines.append(f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Default,,0,0,0,,{text}")

    return lines

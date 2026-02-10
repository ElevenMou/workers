"""Generate ASS (Advanced SubStation Alpha) subtitle files for clip captions.

Two modes are supported:

- **animated**: word-by-word karaoke highlight using ASS ``\\k`` tags.
- **static**: one dialogue line per transcript segment, plain text.

The renderer handles both Whisper transcripts (per-word timing in
``segment["words"]``) and YouTube transcripts (sentence-level only --
word timing is estimated by distributing evenly).
"""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert ``#RRGGBB`` or ``#AARRGGBB`` to ASS ``&HAABBGGRR`` format.

    ASS colours are BGR with an optional alpha byte (00 = opaque).
    """
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}&".upper()
    if len(h) == 8:
        a, r, g, b = h[0:2], h[2:4], h[4:6], h[6:8]
        return f"&H{a}{b}{g}{r}&".upper()
    # Fallback - return white
    return "&H00FFFFFF&"


def _color_name_to_ass(name: str) -> str:
    """Best-effort named-colour → ASS hex.  Falls back to white."""
    _MAP = {
        "white": "&H00FFFFFF&",
        "black": "&H00000000&",
        "red": "&H000000FF&",
        "green": "&H0000FF00&",
        "blue": "&H00FF0000&",
        "yellow": "&H0000FFFF&",
        "cyan": "&H00FFFF00&",
        "magenta": "&H00FF00FF&",
    }
    if name.startswith("#"):
        return _hex_to_ass_color(name)
    return _MAP.get(name.lower(), "&H00FFFFFF&")


# ---------------------------------------------------------------------------
# ASS timestamp formatting
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    """Format seconds to ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Word-timing extraction / estimation
# ---------------------------------------------------------------------------


def _words_from_segment(seg: dict) -> list[dict]:
    """Return a list of ``{word, start, end}`` dicts for *seg*.

    If the segment comes from Whisper it already has a ``words`` key.
    For YouTube transcripts we split the text and distribute timing
    evenly across the segment duration.
    """
    if "words" in seg and seg["words"]:
        return [
            {
                "word": w.get("word", w.get("text", "")).strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
            for w in seg["words"]
            if w.get("word", w.get("text", "")).strip()
        ]

    # Estimate word timing from sentence-level timestamps
    text = seg.get("text", "")
    words = text.split()
    if not words:
        return []

    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    duration = seg_end - seg_start
    per_word = duration / len(words) if len(words) > 0 else duration

    return [
        {
            "word": w,
            "start": seg_start + i * per_word,
            "end": seg_start + (i + 1) * per_word,
        }
        for i, w in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# Segment extraction for a clip time-range
# ---------------------------------------------------------------------------


def extract_clip_segments(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Return only the transcript segments that overlap ``[clip_start, clip_end]``
    with timestamps shifted so ``clip_start`` becomes ``0``.

    YouTube transcripts often have overlapping and long segments.
    This function de-overlaps them and splits long segments into
    shorter sentence-level chunks so captions display cleanly.
    """

    raw: list[dict] = []
    for seg in transcript.get("segments", []):
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # Skip segments entirely outside the clip range
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Clamp to clip boundaries
        clamped_start = max(seg_start, clip_start)
        clamped_end = min(seg_end, clip_end)

        new_seg: dict[str, Any] = {
            "start": clamped_start - clip_start,
            "end": clamped_end - clip_start,
            "text": seg.get("text", ""),
        }

        # Shift word-level timing too
        if "words" in seg and seg["words"]:
            new_words = []
            for w in seg["words"]:
                ws = float(w["start"])
                we = float(w["end"])
                if we <= clip_start or ws >= clip_end:
                    continue
                new_words.append(
                    {
                        "word": w.get("word", w.get("text", "")),
                        "start": max(ws, clip_start) - clip_start,
                        "end": min(we, clip_end) - clip_start,
                    }
                )
            new_seg["words"] = new_words

        raw.append(new_seg)

    # -- De-overlap: trim each segment's end so it doesn't exceed the
    #    next segment's start.  YouTube transcripts commonly overlap.
    raw.sort(key=lambda s: s["start"])
    for i in range(len(raw) - 1):
        if raw[i]["end"] > raw[i + 1]["start"]:
            raw[i]["end"] = raw[i + 1]["start"]

    # -- Split long segments (no word-level timing) into ~8-word chunks
    #    so captions don't show a wall of text at once.
    segments: list[dict] = []
    for seg in raw:
        if "words" in seg and seg["words"]:
            # Whisper segments already have word timing -- keep as-is
            segments.append(seg)
            continue

        words = seg["text"].split()
        if len(words) <= 8:
            segments.append(seg)
            continue

        # Split into chunks of ~8 words, distributing time evenly
        chunk_size = 8
        seg_start = seg["start"]
        seg_duration = seg["end"] - seg["start"]
        total_words = len(words)
        time_per_word = seg_duration / total_words if total_words > 0 else 0

        for ci in range(0, total_words, chunk_size):
            chunk_words = words[ci : ci + chunk_size]
            c_start = seg_start + ci * time_per_word
            c_end = seg_start + min(ci + chunk_size, total_words) * time_per_word
            segments.append(
                {
                    "start": c_start,
                    "end": c_end,
                    "text": " ".join(chunk_words),
                }
            )

    return segments


# ---------------------------------------------------------------------------
# Caption placement (position → ASS alignment + margin)
# ---------------------------------------------------------------------------

_CAPTION_INNER_PAD = 30  # pixels inside the video edge
_CAPTION_OUTER_GAP = 20  # pixels outside the video edge


def _caption_placement(
    position: str,
    font_size: int,
    canvas_h: int,
    vid_y: int | None,
    vid_h: int | None,
) -> tuple[int, int]:
    """Return ``(ass_alignment, margin_v)`` for the given caption *position*.

    When *vid_y* and *vid_h* are provided the positions are relative to
    the video area on the canvas:

    * ``"bottom"``      - inside the video, near its bottom edge.
    * ``"middle"``      - vertically centred within the video.
    * ``"below_video"`` - just below the video area.
    * ``"above_video"`` - just above the video area.

    Without video layout info, positions fall back to simple canvas-relative
    margins.
    """
    if vid_y is not None and vid_h is not None:
        vid_bottom = vid_y + vid_h

        if position == "bottom":
            # Bottom-aligned text, pushed up from canvas bottom so it
            # sits *_CAPTION_INNER_PAD* pixels inside the video's
            # bottom edge.
            alignment = 2
            margin_v = canvas_h - vid_bottom + _CAPTION_INNER_PAD

        elif position == "middle":
            # Top-aligned text placed at the vertical centre of the video.
            alignment = 8
            margin_v = vid_y + (vid_h - font_size) // 2

        elif position == "below_video":
            # Top-aligned text starting just below the video.
            alignment = 8
            margin_v = vid_bottom + _CAPTION_OUTER_GAP

        elif position == "above_video":
            # Bottom-aligned text ending just above the video.
            alignment = 2
            margin_v = canvas_h - vid_y + _CAPTION_OUTER_GAP

        else:
            alignment = 2
            margin_v = 80
    else:
        # Fallback - no video layout info; use simple canvas-relative values.
        alignment = {"bottom": 2, "middle": 5, "top": 8}.get(position, 2)
        margin_v = 80 if position == "bottom" else 40

    return alignment, max(0, margin_v)


# ---------------------------------------------------------------------------
# ASS file generation
# ---------------------------------------------------------------------------


def render_ass(
    segments: list[dict],
    *,
    style: str = "animated",
    font_size: int = 42,
    font_color: str = "white",
    font_family: str = "",
    highlight_color: str = "#FFD700",
    position: str = "bottom",
    max_words_per_line: int = 5,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    vid_y: int | None = None,
    vid_h: int | None = None,
    output_path: str,
) -> str:
    """Generate an ASS subtitle file and return its path.

    Parameters
    ----------
    segments : list[dict]
        Transcript segments already shifted so clip start == 0.
    style : ``"animated"`` or ``"static"``.
    vid_y, vid_h : Video area position/height on the canvas.
        When provided, ``position`` values are relative to the video
        area rather than the full canvas.
    output_path : Destination file path (should end in ``.ass``).
    """
    primary_color = _color_name_to_ass(font_color)
    karaoke_color = _color_name_to_ass(highlight_color)
    fname = font_family or "Arial"

    # Compute ASS alignment and vertical margin.
    # Positions are relative to the *video* area when vid_y/vid_h are given.
    alignment, margin_v = _caption_placement(
        position, font_size, canvas_h, vid_y, vid_h
    )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {canvas_w}\n"
        f"PlayResY: {canvas_h}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fname},{font_size},{primary_color},{karaoke_color},"
        f"&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,3,1,"
        f"{alignment},20,20,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    lines: list[str] = []

    if style == "animated":
        lines = _build_animated_lines(segments, karaoke_color, max_words_per_line)
    else:
        lines = _build_static_lines(segments)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in lines:
            f.write(line + "\n")

    logger.info("Wrote %d caption lines to %s", len(lines), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Animated (karaoke) line builder
# ---------------------------------------------------------------------------


def _build_animated_lines(
    segments: list[dict],
    karaoke_color: str,
    max_words_per_line: int,
) -> list[str]:
    """Build ASS Dialogue lines with ``\\k`` karaoke tags.

    Words are grouped into chunks of *max_words_per_line*.  Within each
    chunk the highlight sweeps across each word.
    """
    lines: list[str] = []

    for seg in segments:
        words = _words_from_segment(seg)
        if not words:
            continue

        # Split into groups of max_words_per_line
        for g_start in range(0, len(words), max_words_per_line):
            group = words[g_start : g_start + max_words_per_line]
            if not group:
                continue

            line_start = group[0]["start"]
            line_end = group[-1]["end"]

            # Build karaoke text: {\kf<cs>}word for each word
            # \kf = karaoke fill, duration in centiseconds
            parts: list[str] = []
            for w in group:
                duration_cs = max(1, int((w["end"] - w["start"]) * 100))
                # Escape braces in the word text
                word_text = w["word"].replace("{", "").replace("}", "")
                parts.append(f"{{\\kf{duration_cs}}}{word_text}")

            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_fmt_ts(line_start)},{_fmt_ts(line_end)},"
                f"Default,,0,0,0,,{text}"
            )

    return lines


# ---------------------------------------------------------------------------
# Static line builder
# ---------------------------------------------------------------------------


def _build_static_lines(segments: list[dict]) -> list[str]:
    """Build plain ASS Dialogue lines -- one per transcript segment."""
    lines: list[str] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # Escape ASS special chars
        text = text.replace("\\", "\\\\").replace("{", "").replace("}", "")
        start = float(seg["start"])
        end = float(seg["end"])
        lines.append(
            f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)}," f"Default,,0,0,0,,{text}"
        )

    return lines

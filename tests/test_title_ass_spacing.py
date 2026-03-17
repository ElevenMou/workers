from __future__ import annotations

import re

from services.clips.constants import TITLE_LINE_HEIGHT_RATIO, TITLE_BAR_V_PAD
from services.clips.ffmpeg_ops import _build_title_ass
from services.clips.layout import wrap_title

_POS_RE = re.compile(r"\\pos\(\d+,(\d+)\)")


def test_title_ass_line_spacing_uses_compact_line_height(tmp_path):
    ass_path = _build_title_ass(
        title_lines=["First line", "Second line", "Third line"],
        duration_seconds=4.0,
        output_path=str(tmp_path / "title.ass"),
        canvas_w=1080,
        canvas_h=1920,
        title_font_size=60,
        title_font_color="#FFFFFF",
        title_font_family="Montserrat-Bold",
        title_align="left",
        title_stroke_width=0,
        title_stroke_color="#000000",
        title_padding_x=16,
        title_text_y=120,
        title_area_x=0,
        title_area_w=1080,
    )

    assert ass_path is not None
    content = (tmp_path / "title.ass").read_text(encoding="utf-8")
    ys: list[int] = []
    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        match = _POS_RE.search(line)
        assert match is not None
        ys.append(int(match.group(1)))

    assert len(ys) == 3
    expected_step = int(60 * TITLE_LINE_HEIGHT_RATIO)
    assert ys[1] - ys[0] == expected_step
    assert ys[2] - ys[1] == expected_step
    # Guardrail: spacing must stay below 1× font size (tight headings).
    assert expected_step <= 60
    # Sanity: vertical pad constant used in bar height must be ≤ 16 (compact).
    assert TITLE_BAR_V_PAD <= 16


def test_wrap_title_uses_full_width():
    """Title wrapping should use the available width, not leave excess space."""
    # At 48px with default ratio 0.52, avg char width ≈ 25px.
    # With 1000px max width → max_chars ≈ 40.
    lines = wrap_title("a " * 30, font_size=48, max_width=1000)
    # Should produce multiple lines, each using most of the available chars.
    assert len(lines) >= 2
    for line in lines[:-1]:
        # Each full line should use at least 60% of max chars (≈24 chars).
        assert len(line) >= 20


def test_wrap_title_font_family_aware():
    """Different fonts produce different wrapping due to width ratios."""
    title = "This is a moderately long title that might wrap differently"
    lines_default = wrap_title(title, font_size=48, max_width=800)
    lines_mono = wrap_title(title, font_size=48, max_width=800, font_family="Space Mono")
    lines_oswald = wrap_title(title, font_size=48, max_width=800, font_family="Oswald")
    # Space Mono is wider → fewer chars per line → more lines.
    # Oswald is narrower → more chars per line → fewer lines.
    assert len(lines_mono) >= len(lines_default)
    assert len(lines_oswald) <= len(lines_default)


def test_wrap_title_max_three_lines():
    """Very long titles are capped at 3 lines with ellipsis."""
    lines = wrap_title("word " * 100, font_size=48, max_width=500)
    assert len(lines) == 3
    assert lines[-1].endswith("...")


def test_wrap_title_handles_edge_cases():
    """Empty, single-word, and newline-containing titles."""
    assert wrap_title("", font_size=48, max_width=500) == []
    assert wrap_title("Hello", font_size=48, max_width=500) == ["Hello"]
    # Embedded newlines are treated as spaces (split on whitespace).
    lines = wrap_title("Hello\nWorld", font_size=48, max_width=500)
    assert lines == ["Hello World"] or lines == ["Hello", "World"]

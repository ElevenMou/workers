from __future__ import annotations

import re

from services.clips.constants import TITLE_LINE_HEIGHT_RATIO
from services.clips.ffmpeg_ops import _build_title_ass

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
    # Guardrail against overly loose title spacing.
    assert expected_step <= int(60 * 1.15)

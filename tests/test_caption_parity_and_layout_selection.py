from __future__ import annotations

import logging
import os
import re
import tempfile
from types import SimpleNamespace

from services.captions.ass_generator import generate_ass_content
from tasks.clips.helpers.captions import _overrides_from_layout, build_caption_ass
from tasks.clips.helpers.layout import EffectiveLayoutSelection, resolve_effective_layout_id


LOGGER = logging.getLogger(__name__)
_DIALOGUE_RE = re.compile(r"^Dialogue:\s*\d+,([^,]+),([^,]+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$")


def _sample_transcript() -> dict:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.4,
                "text": "One two three",
                "words": [
                    {"word": "One", "start": 0.0, "end": 0.8},
                    {"word": "two", "start": 0.8, "end": 1.6},
                    {"word": "three", "start": 1.6, "end": 2.4},
                ],
            }
        ]
    }


def _dialogue_rows(ass_text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in ass_text.splitlines():
        matched = _DIALOGUE_RE.match(line)
        if not matched:
            continue
        rows.append((matched.group(1), matched.group(2), matched.group(3)))
    return rows


def _parse_ass_time(timestamp: str) -> float:
    hours_raw, minutes_raw, seconds_raw = timestamp.split(":")
    seconds_token, centis_raw = seconds_raw.split(".")
    return (
        int(hours_raw) * 3600
        + int(minutes_raw) * 60
        + int(seconds_token)
        + int(centis_raw) / 100.0
    )


def _style_row(ass_text: str) -> list[str]:
    for line in ass_text.splitlines():
        if line.startswith("Style: Default,"):
            return line.split(",")
    raise AssertionError("ASS style row not found")


def test_caption_override_precedence_for_chars_lines_and_delay():
    overrides = _overrides_from_layout(
        {
            "style": "grouped",
            "maxCharsPerCaption": 33,
            "maxCharsPerLine": 44,
            "maxWordsPerLine": 6,
            "linesPerPage": 3,
            "maxLines": 1,
            "lineDelay": 0.12,
        },
        canvas_w=1080,
        canvas_h=1920,
        preset_name="clean",
    )

    assert overrides["max_chars_per_line"] == 33
    assert overrides["max_lines"] == 1
    assert overrides["line_delay"] == 0.12


def test_caption_override_maps_italic_and_underline():
    overrides = _overrides_from_layout(
        {
            "style": "grouped",
            "italic": True,
            "underline": True,
        },
        canvas_w=1080,
        canvas_h=1920,
        preset_name="clean",
    )

    assert overrides["italic"] is True
    assert overrides["underline"] is True


def test_caption_override_preserves_lowercase_font_case():
    overrides = _overrides_from_layout(
        {
            "style": "grouped",
            "fontCase": "lowercase",
        },
        canvas_w=1080,
        canvas_h=1920,
        preset_name="clean",
    )

    assert overrides["font_case"] == "lowercase"
    assert overrides["uppercase"] is False


def test_caption_override_fallbacks_for_legacy_and_words_mapping():
    from_chars_per_line = _overrides_from_layout(
        {
            "style": "grouped",
            "maxCharsPerLine": 41,
            "maxWordsPerLine": 2,
        },
        canvas_w=1080,
        canvas_h=1920,
        preset_name="clean",
    )
    assert from_chars_per_line["max_chars_per_line"] == 41

    from_words_only = _overrides_from_layout(
        {
            "style": "grouped",
            "maxWordsPerLine": 5,
        },
        canvas_w=1080,
        canvas_h=1920,
        preset_name="clean",
    )
    assert from_words_only["max_chars_per_line"] == 30

    # clean preset default delay_between_words is 0.04
    assert abs(from_words_only["line_delay"] - 0.04) < 1e-6


def test_mode_selection_grouped_highlight_and_karaoke():
    transcript = _sample_transcript()

    grouped_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "grouped",
            "word_highlight": False,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    assert grouped_ass.count("Dialogue:") == 1
    assert r"{\kf" not in grouped_ass

    highlight_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "highlight",
            "word_highlight": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    # highlight style produces one event per word (3 words = 3 events)
    assert highlight_ass.count("Dialogue:") == 3
    assert r"{\kf" not in highlight_ass

    karaoke_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "karaoke",
            "word_highlight": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    assert karaoke_ass.count("Dialogue:") == 1
    assert r"{\kf" in karaoke_ass


def test_animation_karaoke_does_not_force_karaoke_mode_without_style_or_word_highlight():
    transcript = _sample_transcript()

    ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "grouped",
            "animation": "karaoke",
            "word_highlight": False,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )

    assert ass.count("Dialogue:") == 1
    assert r"{\kf" not in ass


def test_highlight_events_show_all_words_per_event():
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.5,
                "text": "ONE TWO THREE FOUR FIVE SIX",
                "words": [
                    {"word": "ONE", "start": 0.0, "end": 0.5},
                    {"word": "TWO", "start": 0.5, "end": 1.0},
                    {"word": "THREE", "start": 1.0, "end": 1.5},
                    {"word": "FOUR", "start": 1.5, "end": 2.0},
                    {"word": "FIVE", "start": 2.0, "end": 2.6},
                    {"word": "SIX", "start": 2.6, "end": 3.5},
                ],
            }
        ]
    }

    ass = generate_ass_content(
        transcript,
        "mrbeast",
        overrides={
            "style": "highlight",
            "max_chars_per_line": 10,
            "max_lines": 2,
            "line_delay": 0.0,
        },
    )

    dialogue_texts = [row[2] for row in _dialogue_rows(ass)]
    assert dialogue_texts
    # Highlight style shows all words on every event; multi-line pages use \N
    assert any(r"\N" in text for text in dialogue_texts)


def test_highlight_box_produces_per_word_events():
    ass = generate_ass_content(
        _sample_transcript(),
        "clean",
        overrides={
            "style": "highlight_box",
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )

    rows = _dialogue_rows(ass)
    # 3 words = 3 events for highlight_box
    assert len(rows) == 3
    # Should contain \bord tags for the box effect
    assert any(r"\bord" in row[2] for row in rows)


def test_top_position_anchor_uses_video_bounds_margin():
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.4,
                "text": "One two three",
                "words": [
                    {"word": "One", "start": 0.0, "end": 0.8},
                    {"word": "two", "start": 0.8, "end": 1.6},
                    {"word": "three", "start": 1.6, "end": 2.4},
                ],
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        ass_path = build_caption_ass(
            job_id="job-top-anchor",
            clip_id="clip-top-anchor",
            transcript=transcript,
            cap_cfg={
                "show": True,
                "presetName": "clean",
                "style": "grouped",
                "position": "top",
                "fontFamily": "Montserrat",
                "fontWeight": "bold",
                "maxCharsPerCaption": 36,
                "maxLines": 2,
            },
            start_time=0.0,
            end_time=2.4,
            canvas_w=1080,
            canvas_h=1920,
            vid_y=420,
            vid_h=1080,
            video_aspect_ratio="9:16",
            work_dir=tmp_dir,
            logger=LOGGER,
        )
        assert ass_path is not None
        with open(ass_path, "r", encoding="utf-8") as handle:
            style = _style_row(handle.read())

    # For top position: margin should be anchored to video top + inset = 420 + 80.
    assert style[21] == "500"


def test_italic_and_underline_propagate_to_ass_style():
    transcript = _sample_transcript()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ass_path = build_caption_ass(
            job_id="job-italic-underline",
            clip_id="clip-italic-underline",
            transcript=transcript,
            cap_cfg={
                "show": True,
                "presetName": "clean",
                "style": "grouped",
                "italic": True,
                "underline": True,
                "fontFamily": "Montserrat",
                "fontWeight": "bold",
                "maxCharsPerCaption": 36,
                "maxLines": 2,
            },
            start_time=0.0,
            end_time=2.4,
            canvas_w=1080,
            canvas_h=1920,
            vid_y=420,
            vid_h=1080,
            video_aspect_ratio="9:16",
            work_dir=tmp_dir,
            logger=LOGGER,
        )
        assert ass_path is not None
        with open(ass_path, "r", encoding="utf-8") as handle:
            style = _style_row(handle.read())

    assert style[8] == "-1"
    assert style[9] == "-1"


def test_lowercase_font_case_applies_to_ass_dialogue_text():
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "MiXeD CaSe",
                "words": [
                    {"word": "MiXeD", "start": 0.0, "end": 0.9},
                    {"word": "CaSe", "start": 0.9, "end": 2.0},
                ],
            }
        ]
    }

    ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "grouped",
            "font_case": "lowercase",
            "uppercase": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )

    dialogue_texts = [row[2] for row in _dialogue_rows(ass)]
    assert dialogue_texts
    assert any("mixed case" in text for text in dialogue_texts)
    assert all("MIXED CASE" not in text for text in dialogue_texts)


class _DefaultLayoutQuery:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, _value: str):
        return self

    def eq(self, _key: str, _value):
        return self

    def is_(self, _key: str, _value):
        return self

    def limit(self, _value: int):
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows, error=None)


class _FakeDefaultSupabase:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def table(self, _name: str):
        return _DefaultLayoutQuery(self._rows)


def test_layout_resolution_prefers_requested_over_clip(monkeypatch):
    def fake_candidate(**kwargs):
        if kwargs["source"] == "requested":
            return "requested-layout"
        return "clip-layout"

    monkeypatch.setattr("tasks.clips.helpers.layout._resolve_user_layout_candidate", fake_candidate)

    selection: EffectiveLayoutSelection = resolve_effective_layout_id(
        user_id="user-1",
        job_id="job-1",
        logger=LOGGER,
        requested_layout_id="requested-layout",
        clip_layout_id="clip-layout",
    )

    assert selection.layout_id == "requested-layout"
    assert selection.source == "requested"
    assert selection.should_persist_to_clip is True


def test_layout_resolution_uses_clip_when_requested_invalid(monkeypatch):
    def fake_candidate(**kwargs):
        if kwargs["source"] == "requested":
            return None
        return "clip-layout"

    monkeypatch.setattr("tasks.clips.helpers.layout._resolve_user_layout_candidate", fake_candidate)

    selection: EffectiveLayoutSelection = resolve_effective_layout_id(
        user_id="user-1",
        job_id="job-1",
        logger=LOGGER,
        requested_layout_id="missing-layout",
        clip_layout_id="clip-layout",
    )

    assert selection.layout_id == "clip-layout"
    assert selection.source == "clip"
    assert selection.should_persist_to_clip is False


def test_layout_resolution_falls_back_to_default_then_first_created(monkeypatch):
    monkeypatch.setattr(
        "tasks.clips.helpers.layout._resolve_user_layout_candidate",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "tasks.clips.helpers.layout.supabase",
        _FakeDefaultSupabase(rows=[{"id": "default-layout"}]),
    )
    monkeypatch.setattr(
        "tasks.clips.helpers.layout._first_created_layout_id",
        lambda **_: "first-layout",
    )

    default_selection: EffectiveLayoutSelection = resolve_effective_layout_id(
        user_id="user-1",
        job_id="job-1",
        logger=LOGGER,
        requested_layout_id=None,
        clip_layout_id=None,
    )
    assert default_selection.layout_id == "default-layout"
    assert default_selection.source == "default"

    monkeypatch.setattr(
        "tasks.clips.helpers.layout.supabase",
        _FakeDefaultSupabase(rows=[]),
    )
    first_selection: EffectiveLayoutSelection = resolve_effective_layout_id(
        user_id="user-1",
        job_id="job-1",
        logger=LOGGER,
        requested_layout_id=None,
        clip_layout_id=None,
    )
    assert first_selection.layout_id == "first-layout"
    assert first_selection.source == "first_created"

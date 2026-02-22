from __future__ import annotations

import logging
from types import SimpleNamespace

from services.captions.ass_generator import generate_ass_content
from tasks.clips.helpers.captions import _overrides_from_layout
from tasks.clips.helpers.layout import EffectiveLayoutSelection, resolve_effective_layout_id


LOGGER = logging.getLogger(__name__)


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
    assert overrides["max_lines"] == 3
    assert overrides["line_delay"] == 0.12


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


def test_mode_selection_grouped_word_by_word_and_karaoke():
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

    word_by_word_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "word_by_word",
            "word_highlight": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    assert word_by_word_ass.count("Dialogue:") == 3
    assert r"{\kf" not in word_by_word_ass

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


class _DefaultLayoutQuery:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def select(self, _value: str):
        return self

    def eq(self, _key: str, _value):
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

from __future__ import annotations

from tasks.videos.split_video import (
    _build_part_windows,
    _combine_removal_ranges,
    _format_part_title,
    _remap_transcript_to_clean_timeline,
)


def test_build_part_windows_keeps_short_final_tail():
    windows = _build_part_windows(duration_seconds=125.0, segment_length_seconds=60)

    assert windows == [
        {"index": 1, "start": 0.0, "end": 60.0},
        {"index": 2, "start": 60.0, "end": 120.0},
        {"index": 3, "start": 120.0, "end": 125.0},
    ]


def test_build_part_windows_snaps_to_nearby_transcript_boundary():
    windows = _build_part_windows(
        duration_seconds=125.0,
        segment_length_seconds=60,
        transcript={
            "segments": [
                {"start": 0.0, "end": 58.0, "text": "Opening thought"},
                {"start": 58.0, "end": 118.0, "text": "Main section"},
                {"start": 118.0, "end": 125.0, "text": "Closing beat"},
            ]
        },
    )

    assert windows == [
        {"index": 1, "start": 0.0, "end": 58.0},
        {"index": 2, "start": 58.0, "end": 118.0},
        {"index": 3, "start": 118.0, "end": 125.0},
    ]


def test_build_part_windows_keeps_short_final_tail_even_if_later_boundary_exists():
    windows = _build_part_windows(
        duration_seconds=124.0,
        segment_length_seconds=60,
        transcript={
            "segments": [
                {"start": 0.0, "end": 60.0, "text": "First part"},
                {"start": 60.0, "end": 124.0, "text": "Second part"},
            ]
        },
    )

    assert windows == [
        {"index": 1, "start": 0.0, "end": 60.0},
        {"index": 2, "start": 60.0, "end": 120.0},
        {"index": 3, "start": 120.0, "end": 124.0},
    ]


def test_combine_removal_ranges_prefers_confident_overlapping_ranges():
    heuristic = [
        {
            "kind": "intro",
            "start": 0.0,
            "end": 12.0,
            "confidence": 0.8,
            "source": "heuristic",
        }
    ]
    ai = [
        {
            "kind": "intro",
            "start": 0.0,
            "end": 11.5,
            "confidence": 0.74,
            "source": "ai",
        }
    ]

    intervals, diagnostics = _combine_removal_ranges(
        heuristic_ranges=heuristic,
        ai_ranges=ai,
        duration_seconds=300.0,
    )

    assert intervals == [(0.0, 12.0)]
    assert diagnostics["accepted_ranges"][0]["sources"] == ["ai", "heuristic"]


def test_combine_removal_ranges_skips_malformed_ai_candidates():
    intervals, diagnostics = _combine_removal_ranges(
        heuristic_ranges=[],
        ai_ranges=[
            "bad-payload",
            {"kind": "ad", "start": "oops", "end": 15.0, "confidence": 0.95},
            {
                "kind": "ad",
                "start": "12.5",
                "end": "22.5",
                "confidence": "0.91",
                "source": "ai",
            },
        ],
        duration_seconds=120.0,
    )

    assert intervals == [(12.5, 22.5)]
    assert diagnostics["accepted_ranges"] == [
        {
            "kind": "ad",
            "start": 12.5,
            "end": 22.5,
            "confidence": 0.91,
            "sources": ["ai"],
            "reasons": [],
        }
    ]


def test_remap_transcript_to_clean_timeline_shifts_preserved_segments():
    transcript = {
        "source": "youtube",
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Intro"},
            {"start": 5.0, "end": 10.0, "text": "Keep this"},
            {"start": 12.0, "end": 18.0, "text": "And this too"},
        ],
    }
    timeline_map = [
        {
            "source_start": 5.0,
            "source_end": 10.0,
            "output_start": 0.0,
            "output_end": 5.0,
        },
        {
            "source_start": 12.0,
            "source_end": 18.0,
            "output_start": 5.0,
            "output_end": 11.0,
        },
    ]

    remapped = _remap_transcript_to_clean_timeline(
        transcript=transcript,
        timeline_map=timeline_map,
    )

    assert remapped["segments"] == [
        {"start": 0.0, "end": 5.0, "text": "Keep this"},
        {"start": 5.0, "end": 11.0, "text": "And this too"},
    ]


def test_format_part_title_strips_duplicate_part_prefix():
    assert _format_part_title(3, "Part 3 - Mejores trucos", "Video base") == "Part 3 - Mejores trucos"
    assert _format_part_title(4, "", "Video base") == "Part 4 - Video base"

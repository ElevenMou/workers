from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

if "whisper" not in sys.modules:
    whisper_stub = ModuleType("whisper")
    whisper_stub.load_model = lambda *_args, **_kwargs: SimpleNamespace(
        transcribe=lambda *_a, **_k: {"text": "", "segments": [], "language": "en"}
    )
    sys.modules["whisper"] = whisper_stub

from tasks.clips import custom as custom_task_module
from tasks.clips import generate as generate_task_module
from tasks.clips.helpers import lifecycle as lifecycle_helper
from tasks.clips.helpers import smart_cleanup as smart_cleanup
from tasks.clips.helpers import source_video as source_video_helper


def _transcript_from_words(words: list[tuple[str, float, float]]) -> dict:
    return {
        "source": "whisper",
        "language": "en",
        "segments": [
            {
                "start": float(words[0][1]) if words else 0.0,
                "end": float(words[-1][2]) if words else 0.0,
                "text": " ".join(token for token, _, _ in words),
                "words": [
                    {"word": token, "start": float(start), "end": float(end)}
                    for token, start, end in words
                ],
            }
        ],
    }


def _flatten_cleaned_words(cleaned_transcript: dict) -> list[str]:
    words: list[str] = []
    for segment in cleaned_transcript.get("segments", []):
        for word in segment.get("words", []):
            token = str(word.get("word", "")).strip().lower()
            if token:
                words.append(token)
    return words


def test_generate_warns_for_low_source_resolution_when_output_is_high(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        generate_task_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    warned = generate_task_module._warn_low_source_resolution_for_high_output(
        job_id="job-1",
        video_id="video-1",
        output_quality="high",
        source_width=640,
        source_height=360,
        source_strategy="downloaded_now",
    )

    assert warned is True
    assert len(warnings) == 1
    assert "low resolution" in warnings[0]


def test_custom_does_not_warn_for_non_high_output_quality(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        custom_task_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    warned = custom_task_module._warn_low_source_resolution_for_high_output(
        job_id="job-2",
        video_id="video-2",
        output_quality="medium",
        source_width=640,
        source_height=360,
        source_strategy="downloaded_now",
    )

    assert warned is False
    assert warnings == []


def test_plan_expands_requested_window_to_word_boundaries(monkeypatch):
    monkeypatch.setattr(
        smart_cleanup,
        "_load_multilingual_stopwords",
        lambda _languages: frozenset(),
    )

    transcript = _transcript_from_words(
        [
            ("alpha", 0.0, 0.4),
            ("beta", 0.5, 1.0),
            ("gamma", 1.2, 1.6),
        ]
    )
    plan = smart_cleanup.plan_balanced_smart_cleanup(
        transcript=transcript,
        start_time=0.2,
        end_time=0.8,
    )

    summary = plan["summary"]
    assert summary["requested_window_start"] == pytest.approx(0.2)
    assert summary["requested_window_end"] == pytest.approx(0.8)
    assert summary["effective_window_start"] == pytest.approx(0.0)
    assert summary["effective_window_end"] == pytest.approx(1.0)
    assert _flatten_cleaned_words(plan["cleaned_transcript"]) == ["alpha", "beta"]


def test_removal_and_keep_boundaries_never_cut_inside_words(monkeypatch):
    monkeypatch.setattr(
        smart_cleanup,
        "_load_multilingual_stopwords",
        lambda _languages: frozenset(),
    )

    transcript = _transcript_from_words(
        [
            ("hello", 0.0, 0.5),
            ("um", 0.75, 0.9),
            ("world", 1.15, 1.6),
        ]
    )
    plan = smart_cleanup.plan_balanced_smart_cleanup(
        transcript=transcript,
        start_time=0.0,
        end_time=1.6,
    )

    source_words = transcript["segments"][0]["words"]
    boundaries = [
        *[edge for interval in plan["keep_intervals"] for edge in interval],
        *[edge for interval in plan["removal_intervals"] for edge in interval],
    ]

    for boundary in boundaries:
        for word in source_words:
            assert not (word["start"] + 1e-6 < boundary < word["end"] - 1e-6)

    assert "um" not in _flatten_cleaned_words(plan["cleaned_transcript"])


def test_cleaned_transcript_drops_partial_word_overlaps():
    cleaned_transcript, dropped_partial_words = smart_cleanup._build_cleaned_transcript(
        source_words=[
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 1.2, "end": 1.6},
        ],
        timeline_map=[
            {
                "source_start": 0.5,
                "source_end": 2.0,
                "output_start": 0.0,
                "output_end": 1.5,
            }
        ],
    )

    assert dropped_partial_words == 1
    assert _flatten_cleaned_words(cleaned_transcript) == ["world"]


def test_plan_requires_whisper_word_timing():
    transcript_without_word_timing = {
        "source": "youtube",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
    }

    with pytest.raises(RuntimeError, match="Whisper word-level timings"):
        smart_cleanup.plan_balanced_smart_cleanup(
            transcript=transcript_without_word_timing,
            start_time=0.0,
            end_time=1.0,
        )


def test_word_level_caption_styles_retranscribe_when_whisper_transcript_has_no_words():
    transcript = {
        "source": "whisper",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
    }

    assert generate_task_module.needs_whisper_retranscription(transcript, "karaoke") is True
    assert generate_task_module.needs_whisper_retranscription(transcript, "highlight") is True
    assert generate_task_module.needs_whisper_retranscription(transcript, "grouped") is False


def test_word_level_caption_styles_retranscribe_when_transcript_is_missing():
    assert generate_task_module.needs_whisper_retranscription(None, "highlight_box") is True
    assert generate_task_module.needs_whisper_retranscription(None, "grouped") is False


class _StopAfterCleanup(RuntimeError):
    pass


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeSupabaseQuery:
    def __init__(self, table_name: str, clip_payload: dict, video_payload: dict):
        self.table_name = table_name
        self.clip_payload = clip_payload
        self.video_payload = video_payload
        self._mode = "select"
        self._select_columns = ""

    def select(self, columns: str):
        self._mode = "select"
        self._select_columns = columns
        return self

    def update(self, _payload: dict):
        self._mode = "update"
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table_name == "clips" and self._mode == "select":
            if "videos(*)" in self._select_columns:
                return _FakeResponse(self.clip_payload)
            return _FakeResponse([])
        if self.table_name == "videos" and self._mode == "select":
            return _FakeResponse(self.video_payload)
        return _FakeResponse({})


class _FakeSupabase:
    def __init__(self, clip_payload: dict, video_payload: dict):
        self.clip_payload = clip_payload
        self.video_payload = video_payload

    def table(self, table_name: str):
        return _FakeSupabaseQuery(table_name, self.clip_payload, self.video_payload)


def test_generate_flow_reuses_existing_word_timing_for_smart_cleanup(monkeypatch, tmp_path: Path):
    transcript_calls = {"count": 0}

    clip_payload = {
        "id": "clip-1",
        "video_id": "video-1",
        "layout_id": None,
        "start_time": 10.2,
        "end_time": 11.8,
        "title": "Clip title",
        # Existing word-timed transcript should be reused for Smart Cleanup.
        "transcript": _transcript_from_words([("old", 10.2, 10.7), ("words", 10.8, 11.4)]),
        "videos": {
            "raw_video_path": "C:/tmp/video.mp4",
            "duration_seconds": 120,
            "url": "https://example.com/video",
            "transcript": None,
        },
    }

    monkeypatch.setattr(
        generate_task_module,
        "supabase",
        _FakeSupabase(clip_payload=clip_payload, video_payload={}),
    )
    monkeypatch.setattr(generate_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "_is_latest_generate_job_for_clip", lambda **_k: True)
    monkeypatch.setattr(
        generate_task_module,
        "best_effort_cleanup_uploaded_artifacts",
        lambda **_k: None,
    )
    monkeypatch.setattr(generate_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle_helper, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "needs_whisper_retranscription", lambda *_a, **_k: False)
    monkeypatch.setattr(generate_task_module, "get_credit_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(generate_task_module, "get_team_wallet_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(source_video_helper, "probe_video_size", lambda *_a, **_k: (1080, 1920))
    monkeypatch.setattr(
        generate_task_module,
        "compute_video_position",
        lambda *_a, **_k: (1080, 1440, 0, 240),
    )
    monkeypatch.setattr(generate_task_module.os.path, "isfile", lambda *_a, **_k: True)
    monkeypatch.setattr(
        generate_task_module,
        "resolve_effective_layout_id",
        lambda **_k: SimpleNamespace(layout_id=None, should_persist_to_clip=False),
    )
    monkeypatch.setattr(
        generate_task_module,
        "load_layout_overrides",
        lambda **_k: SimpleNamespace(
            bg_style="blur",
            bg_color="#000000",
            blur_strength=20,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle", "canvasAspectRatio": "9:16", "videoScaleMode": "fit"},
            {
                "show": False,
                "fontSize": 42,
                "fontColor": "#FFFFFF",
                "fontFamily": "Montserrat",
                "align": "center",
                "strokeWidth": 0,
                "strokeColor": "#000000",
                "barEnabled": False,
                "barColor": "#000000",
                "paddingX": 16,
                "positionY": "top",
            },
            {"show": False},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "storagePath": "", "widthPx": 200, "x": 0, "y": 0},
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "maybe_download_layout_background_image",
        lambda **_k: ("blur", None),
    )
    monkeypatch.setattr(
        generate_task_module,
        "create_work_dir",
        lambda name: str((tmp_path / name).mkdir(parents=True, exist_ok=True) or (tmp_path / name)),
    )
    monkeypatch.setattr(
        generate_task_module,
        "ClipGenerator",
        lambda **_k: SimpleNamespace(generate=lambda **_kwargs: {}),
    )

    def _fake_transcribe(**_kwargs):
        transcript_calls["count"] += 1
        return _transcript_from_words([("fresh", 10.2, 10.7), ("timing", 10.8, 11.4)])

    monkeypatch.setattr(generate_task_module, "transcribe_clip_window_with_whisper", _fake_transcribe)
    monkeypatch.setattr(
        generate_task_module,
        "apply_balanced_smart_cleanup",
        lambda **_k: (_ for _ in ()).throw(_StopAfterCleanup("stop after cleanup call")),
    )

    with pytest.raises(_StopAfterCleanup):
        generate_task_module.generate_clip_task(
            {
                "jobId": "job-1",
                "clipId": "clip-1",
                "userId": "user-1",
                "smartCleanupEnabled": True,
            }
        )

    assert transcript_calls["count"] == 0


def test_custom_flow_reuses_existing_word_timing_for_smart_cleanup(monkeypatch, tmp_path: Path):
    transcript_calls = {"count": 0}

    existing_transcript = _transcript_from_words([("existing", 5.0, 5.4), ("words", 5.5, 6.0)])
    monkeypatch.setattr(
        custom_task_module,
        "supabase",
        _FakeSupabase(
            clip_payload={},
            video_payload={"transcript": existing_transcript},
        ),
    )
    monkeypatch.setattr(custom_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        custom_task_module,
        "best_effort_cleanup_uploaded_artifacts",
        lambda **_k: None,
    )
    monkeypatch.setattr(custom_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(custom_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle_helper, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(custom_task_module, "update_video_status", lambda *_a, **_k: None)
    monkeypatch.setattr(custom_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(custom_task_module, "get_credit_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(custom_task_module, "get_team_wallet_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(custom_task_module, "reserve_credits", lambda **_k: "res-1")
    monkeypatch.setattr(custom_task_module, "capture_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(custom_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(
        custom_task_module,
        "resolve_effective_layout_id",
        lambda **_k: SimpleNamespace(layout_id=None, should_persist_to_clip=False),
    )
    monkeypatch.setattr(
        custom_task_module,
        "load_layout_overrides",
        lambda **_k: SimpleNamespace(
            bg_style="blur",
            bg_color="#000000",
            blur_strength=20,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(
        custom_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle", "canvasAspectRatio": "9:16", "videoScaleMode": "fit"},
            {
                "show": False,
                "fontSize": 42,
                "fontColor": "#FFFFFF",
                "fontFamily": "Montserrat",
                "align": "center",
                "strokeWidth": 0,
                "strokeColor": "#000000",
                "barEnabled": False,
                "barColor": "#000000",
                "paddingX": 16,
                "positionY": "top",
            },
            {"show": False},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "storagePath": "", "widthPx": 200, "x": 0, "y": 0},
        ),
    )
    monkeypatch.setattr(
        custom_task_module,
        "maybe_download_layout_background_image",
        lambda **_k: ("blur", None),
    )
    monkeypatch.setattr(
        custom_task_module,
        "create_work_dir",
        lambda name: str((tmp_path / name).mkdir(parents=True, exist_ok=True) or (tmp_path / name)),
    )
    monkeypatch.setattr(source_video_helper.os.path, "isfile", lambda *_a, **_k: True)
    monkeypatch.setattr(source_video_helper, "probe_video_size", lambda *_a, **_k: (1080, 1920))

    class _FakeDownloader:
        def __init__(self, work_dir: str):
            self.work_dir = work_dir

        def download(self, _url: str, _video_id: str):
            return {
                "path": "C:/tmp/video.mp4",
                "duration": 120,
                "title": "Video",
                "thumbnail": "",
                "platform": "youtube",
                "external_id": "abc123",
            }

        def get_youtube_transcript(self, _external_id: str):
            return None

        def extract_audio(self, _video_path: str):
            return "C:/tmp/audio.mp3"

    monkeypatch.setattr(custom_task_module, "VideoDownloader", _FakeDownloader)

    def _fake_transcribe(**_kwargs):
        transcript_calls["count"] += 1
        return _transcript_from_words([("fresh", 5.0, 5.4), ("timing", 5.5, 6.0)])

    monkeypatch.setattr(custom_task_module, "transcribe_clip_window_with_whisper", _fake_transcribe)
    monkeypatch.setattr(
        custom_task_module,
        "apply_balanced_smart_cleanup",
        lambda **_k: (_ for _ in ()).throw(_StopAfterCleanup("stop after cleanup call")),
    )

    with pytest.raises(_StopAfterCleanup):
        custom_task_module.custom_clip_task(
            {
                "jobId": "job-2",
                "videoId": "video-1",
                "clipId": "clip-2",
                "userId": "user-1",
                "url": "https://example.com/video",
                "startTime": 5.0,
                "endTime": 6.2,
                "title": "Custom clip",
                "smartCleanupEnabled": True,
            }
        )

    assert transcript_calls["count"] == 0


def test_custom_partial_whisper_fallback_returns_transcript(monkeypatch, tmp_path: Path):
    transcript_calls = {"count": 0}

    monkeypatch.setattr(
        custom_task_module,
        "supabase",
        _FakeSupabase(
            clip_payload={},
            video_payload={"transcript": None},
        ),
    )
    monkeypatch.setattr(custom_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        custom_task_module,
        "best_effort_cleanup_uploaded_artifacts",
        lambda **_k: None,
    )
    monkeypatch.setattr(custom_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(custom_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle_helper, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(custom_task_module, "update_video_status", lambda *_a, **_k: None)
    monkeypatch.setattr(custom_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(custom_task_module, "get_credit_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(custom_task_module, "get_team_wallet_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(custom_task_module, "reserve_credits", lambda **_k: "res-1")
    monkeypatch.setattr(custom_task_module, "capture_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(custom_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(
        custom_task_module,
        "resolve_effective_layout_id",
        lambda **_k: SimpleNamespace(layout_id=None, should_persist_to_clip=False),
    )
    monkeypatch.setattr(
        custom_task_module,
        "load_layout_overrides",
        lambda **_k: SimpleNamespace(
            bg_style="blur",
            bg_color="#000000",
            blur_strength=20,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(
        custom_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle", "canvasAspectRatio": "9:16", "videoScaleMode": "fit"},
            {
                "show": False,
                "fontSize": 42,
                "fontColor": "#FFFFFF",
                "fontFamily": "Montserrat",
                "align": "center",
                "strokeWidth": 0,
                "strokeColor": "#000000",
                "barEnabled": False,
                "barColor": "#000000",
                "paddingX": 16,
                "positionY": "top",
            },
            {"show": False},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "type": "image", "storagePath": "", "durationSeconds": 3.0},
            {"enabled": False, "storagePath": "", "widthPx": 200, "x": 0, "y": 0},
        ),
    )
    monkeypatch.setattr(
        custom_task_module,
        "create_work_dir",
        lambda name: str((tmp_path / name).mkdir(parents=True, exist_ok=True) or (tmp_path / name)),
    )
    monkeypatch.setattr(source_video_helper.os.path, "isfile", lambda *_a, **_k: True)
    monkeypatch.setattr(source_video_helper, "probe_video_size", lambda *_a, **_k: (1080, 1920))

    class _FakeDownloader:
        def __init__(self, work_dir: str):
            self.work_dir = work_dir

        def download(self, _url: str, _video_id: str):
            return {
                "path": "C:/tmp/video.mp4",
                "duration": 120,
                "title": "Video",
                "thumbnail": "",
                "platform": "youtube",
                "external_id": "abc123",
            }

        def get_youtube_transcript(self, _external_id: str):
            return None

        def extract_audio(self, _video_path: str):
            return "C:/tmp/audio.mp3"

    monkeypatch.setattr(custom_task_module, "VideoDownloader", _FakeDownloader)

    def _fake_transcribe(**_kwargs):
        transcript_calls["count"] += 1
        return _transcript_from_words([("fallback", 5.0, 5.4), ("words", 5.5, 6.0)])

    def _fake_resolve_source_transcript(*, whisper_fallback, **_kwargs):
        transcript, is_full_transcript = whisper_fallback("en")
        return SimpleNamespace(
            transcript=transcript,
            is_full_transcript=is_full_transcript,
        )

    monkeypatch.setattr(custom_task_module, "transcribe_clip_window_with_whisper", _fake_transcribe)
    monkeypatch.setattr(custom_task_module, "resolve_source_transcript", _fake_resolve_source_transcript)
    monkeypatch.setattr(
        custom_task_module,
        "maybe_download_layout_background_image",
        lambda **_k: (_ for _ in ()).throw(_StopAfterCleanup("stop after transcript fallback")),
    )

    with pytest.raises(_StopAfterCleanup):
        custom_task_module.custom_clip_task(
            {
                "jobId": "job-3",
                "videoId": "video-1",
                "clipId": "clip-3",
                "userId": "user-1",
                "url": "https://example.com/video",
                "startTime": 5.0,
                "endTime": 6.2,
                "title": "Custom clip",
                "smartCleanupEnabled": False,
            }
        )

    assert transcript_calls["count"] == 1

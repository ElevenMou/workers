from __future__ import annotations

from types import SimpleNamespace

import pytest

import tasks.clips.generate as generate_task_module


class _FakeResponse:
    def __init__(self, data):
        self.data = data
        self.error = None


class _FakeClipsQuery:
    def __init__(self, store: "_FakeSupabase"):
        self.store = store
        self.clip_row = store.clip_row
        self._mode = "select"
        self._payload = None

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        return self

    def update(self, payload, *_args, **_kwargs):
        self._mode = "update"
        self._payload = payload
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        if self._mode == "select":
            return _FakeResponse(self.clip_row)
        if isinstance(self._payload, dict):
            self.store.clip_updates.append(dict(self._payload))
            self.clip_row.update(self._payload)
        return _FakeResponse([self.clip_row])


class _FakeSupabase:
    def __init__(self, clip_row: dict):
        self.clip_row = clip_row
        self.clip_updates: list[dict] = []

    def table(self, name: str):
        if name != "clips":
            raise AssertionError(f"Unexpected table access: {name}")
        return _FakeClipsQuery(self)


def test_generate_clip_uses_split_source_override_for_resolution(monkeypatch, tmp_path):
    clip_row = {
        "id": "clip-1",
        "video_id": "video-1",
        "title": "Part 1 - Hola",
        "layout_id": None,
        "start_time": 0.0,
        "end_time": 60.0,
        "status": "pending",
        "transcript": {"segments": []},
        "source_video_storage_path_override": "raw/video-1__source_split_cleaned.mov",
        "videos": {
            "url": "https://www.youtube.com/watch?v=abc123",
            "raw_video_path": str(tmp_path / "original.mp4"),
            "raw_video_storage_path": "raw/video-1__source_h1080.mkv",
            "transcript": {"segments": []},
            "duration_seconds": 300,
        },
    }
    recorded: dict[str, object] = {}

    def _fake_resolve_source_video(**kwargs):
        recorded.update(kwargs)
        raise RuntimeError("stop after source resolution")

    monkeypatch.setattr(generate_task_module, "supabase", _FakeSupabase(clip_row))
    monkeypatch.setattr(generate_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(generate_task_module, "reserve_credits", lambda **_k: "reservation-1")
    monkeypatch.setattr(generate_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_is_latest_generate_job_for_clip", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "best_effort_cleanup_uploaded_artifacts", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_clip_job_progress", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "create_work_dir", lambda _name: str(tmp_path))
    monkeypatch.setattr(generate_task_module, "resolve_effective_layout_id", lambda **_k: SimpleNamespace(layout_id=None, should_persist_to_clip=False))
    monkeypatch.setattr(
        generate_task_module,
        "load_layout_overrides",
        lambda **_k: SimpleNamespace(
            bg_style="blur",
            bg_color="#000000",
            blur_strength=0,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={"show": False},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(generate_task_module, "resolve_effective_output_quality", lambda **_k: "medium")
    monkeypatch.setattr(
        generate_task_module,
        "resolve_quality_controls",
        lambda **_k: SimpleNamespace(
            effective_source_max_height=None,
            prefer_fresh_source_download=False,
            allow_upload_reencode=False,
            smart_cleanup_crf=21,
            smart_cleanup_preset="medium",
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle"},
            {"show": True},
            {"show": False},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr(generate_task_module, "maybe_download_layout_background_image", lambda **_k: ("blur", None))
    monkeypatch.setattr(generate_task_module, "maybe_download_media_files", lambda **_k: (None, None, None))
    monkeypatch.setattr(generate_task_module, "resolve_source_video", _fake_resolve_source_video)

    with pytest.raises(RuntimeError, match="stop after source resolution"):
        generate_task_module.generate_clip_task(
            {
                "jobId": "job-1",
                "clipId": "clip-1",
                "userId": "user-1",
                "generationCredits": 3,
                "subscriptionTier": "basic",
                "sourceProfile": "source_h1080",
            }
        )

    assert recorded["initial_raw_video_path"] is None
    assert recorded["initial_raw_video_storage_path"] == "raw/video-1__source_split_cleaned.mov"
    assert recorded["source_profile"] == "source_h1080_override"


def test_generate_clip_renders_deferred_split_source_only_when_generation_starts(
    monkeypatch,
    tmp_path,
):
    clip_row = {
        "id": "clip-1",
        "video_id": "video-1",
        "title": "Part 2 - Hola",
        "layout_id": None,
        "start_time": 60.0,
        "end_time": 120.0,
        "status": "pending",
        "transcript": {
            "segments": [
                {"start": 60.0, "end": 65.0, "text": "Shift me"},
            ],
            "metadata": {
                "batch_split_deferred_render": {
                    "strategy": "source_keep_intervals",
                    "keep_intervals": [[70.0, 100.0], [110.0, 140.0]],
                    "output_duration_seconds": 60.0,
                    "cleaned_window_start": 60.0,
                    "cleaned_window_end": 120.0,
                    "transcript_offset_seconds": 60.0,
                }
            },
        },
        "source_video_storage_path_override": None,
        "videos": {
            "url": "https://www.youtube.com/watch?v=abc123",
            "raw_video_path": str(tmp_path / "original.mp4"),
            "raw_video_storage_path": "raw/video-1__source_h1080.mkv",
            "transcript": {"segments": []},
            "duration_seconds": 300,
        },
    }
    recorded: dict[str, object] = {}

    def _fake_resolve_source_video(**kwargs):
        recorded["resolve_source_video"] = dict(kwargs)
        return SimpleNamespace(
            video_path=str(tmp_path / "original.mp4"),
            width=1920,
            height=1080,
            strategy="reused_existing",
            wait_seconds=0.0,
            download_seconds=0.0,
        )

    def _fake_render_condensed_video_from_keep_intervals(**kwargs):
        recorded["deferred_render"] = dict(kwargs)
        return 60.0

    def _fake_build_caption_ass(**kwargs):
        recorded["caption_call"] = dict(kwargs)
        raise RuntimeError("stop after deferred split prep")

    monkeypatch.setattr(generate_task_module, "supabase", _FakeSupabase(clip_row))
    monkeypatch.setattr(generate_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(generate_task_module, "reserve_credits", lambda **_k: "reservation-1")
    monkeypatch.setattr(generate_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_is_latest_generate_job_for_clip", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "best_effort_cleanup_uploaded_artifacts", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_clip_job_progress", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "create_work_dir", lambda _name: str(tmp_path))
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
            blur_strength=0,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={"show": False},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(generate_task_module, "resolve_effective_output_quality", lambda **_k: "medium")
    monkeypatch.setattr(
        generate_task_module,
        "resolve_quality_controls",
        lambda **_k: SimpleNamespace(
            effective_source_max_height=None,
            prefer_fresh_source_download=False,
            allow_upload_reencode=False,
            smart_cleanup_crf=21,
            smart_cleanup_preset="medium",
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle"},
            {"show": True},
            {"show": False},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr(generate_task_module, "maybe_download_layout_background_image", lambda **_k: ("blur", None))
    monkeypatch.setattr(generate_task_module, "maybe_download_media_files", lambda **_k: (None, None, None))
    monkeypatch.setattr(generate_task_module, "resolve_source_video", _fake_resolve_source_video)
    monkeypatch.setattr(
        generate_task_module,
        "render_condensed_video_from_keep_intervals",
        _fake_render_condensed_video_from_keep_intervals,
    )
    monkeypatch.setattr(generate_task_module, "build_caption_ass", _fake_build_caption_ass)

    with pytest.raises(RuntimeError, match="stop after deferred split prep"):
        generate_task_module.generate_clip_task(
            {
                "jobId": "job-1",
                "clipId": "clip-1",
                "userId": "user-1",
                "generationCredits": 3,
                "subscriptionTier": "basic",
                "sourceProfile": "source_h1080",
            }
        )

    resolve_call = recorded["resolve_source_video"]
    assert resolve_call["initial_raw_video_path"] == str(tmp_path / "original.mp4")
    assert resolve_call["initial_raw_video_storage_path"] == "raw/video-1__source_h1080.mkv"
    assert resolve_call["source_profile"] == "source_h1080"

    deferred_render = recorded["deferred_render"]
    assert deferred_render["input_video_path"] == str(tmp_path / "original.mp4")
    assert deferred_render["keep_intervals"] == [(70.0, 100.0), (110.0, 140.0)]

    caption_call = recorded["caption_call"]
    assert caption_call["start_time"] == 0.0
    assert caption_call["end_time"] == 60.0
    assert caption_call["transcript"]["segments"][0]["start"] == 0.0
    assert caption_call["transcript"]["segments"][0]["end"] == 5.0


def test_generate_clip_preserves_deferred_split_metadata_when_retranscribing(
    monkeypatch,
    tmp_path,
):
    clip_row = {
        "id": "clip-1",
        "video_id": "video-1",
        "title": "Part 2 - Hola",
        "layout_id": None,
        "start_time": 60.0,
        "end_time": 120.0,
        "status": "pending",
        "transcript": {
            "segments": [{"start": 60.0, "end": 65.0, "text": "Shift me"}],
            "metadata": {
                "batch_split_deferred_render": {
                    "strategy": "source_keep_intervals",
                    "keep_intervals": [[70.0, 100.0], [110.0, 140.0]],
                    "output_duration_seconds": 60.0,
                    "cleaned_window_start": 60.0,
                    "cleaned_window_end": 120.0,
                    "transcript_offset_seconds": 60.0,
                }
            },
        },
        "source_video_storage_path_override": None,
        "videos": {
            "url": "https://www.youtube.com/watch?v=abc123",
            "raw_video_path": str(tmp_path / "original.mp4"),
            "raw_video_storage_path": "raw/video-1__source_h1080.mkv",
            "transcript": {"segments": []},
            "duration_seconds": 300,
        },
    }
    fake_supabase = _FakeSupabase(clip_row)

    monkeypatch.setattr(generate_task_module, "supabase", fake_supabase)
    monkeypatch.setattr(generate_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(generate_task_module, "reserve_credits", lambda **_k: "reservation-1")
    monkeypatch.setattr(generate_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_is_latest_generate_job_for_clip", lambda **_k: True)
    monkeypatch.setattr(generate_task_module, "_best_effort_mark_failed", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "best_effort_cleanup_uploaded_artifacts", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_clip_job_progress", lambda **_k: None)
    monkeypatch.setattr(generate_task_module, "update_job_status", lambda *_a, **_k: None)
    monkeypatch.setattr(generate_task_module, "create_work_dir", lambda _name: str(tmp_path))
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
            blur_strength=0,
            output_quality="medium",
            layout_video={},
            layout_title={},
            layout_captions={"show": True},
            layout_intro={},
            layout_outro={},
            layout_overlay={},
            bg_image_storage_path=None,
        ),
    )
    monkeypatch.setattr(generate_task_module, "resolve_effective_output_quality", lambda **_k: "medium")
    monkeypatch.setattr(
        generate_task_module,
        "resolve_quality_controls",
        lambda **_k: SimpleNamespace(
            effective_source_max_height=None,
            prefer_fresh_source_download=False,
            allow_upload_reencode=False,
            smart_cleanup_crf=21,
            smart_cleanup_preset="medium",
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "merge_layout_configs",
        lambda *_a, **_k: (
            {"widthPct": 100, "positionY": "middle"},
            {"show": True},
            {"show": True, "style": "karaoke"},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr(generate_task_module, "maybe_download_layout_background_image", lambda **_k: ("blur", None))
    monkeypatch.setattr(generate_task_module, "maybe_download_media_files", lambda **_k: (None, None, None))
    monkeypatch.setattr(
        generate_task_module,
        "resolve_source_video",
        lambda **_kwargs: SimpleNamespace(
            video_path=str(tmp_path / "original.mp4"),
            width=1920,
            height=1080,
            strategy="reused_existing",
            wait_seconds=0.0,
            download_seconds=0.0,
        ),
    )
    monkeypatch.setattr(
        generate_task_module,
        "render_condensed_video_from_keep_intervals",
        lambda **_kwargs: 60.0,
    )
    monkeypatch.setattr(generate_task_module, "resolve_caption_style_mode", lambda _cfg: "word_highlight")
    monkeypatch.setattr(generate_task_module, "needs_whisper_retranscription", lambda *_a, **_k: True)
    monkeypatch.setattr(
        generate_task_module,
        "transcribe_clip_window_with_whisper",
        lambda **_kwargs: {"segments": [{"start": 0.0, "end": 1.0, "text": "Fresh"}]},
    )
    monkeypatch.setattr(
        generate_task_module,
        "build_caption_ass",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after transcript save")),
    )

    with pytest.raises(RuntimeError, match="stop after transcript save"):
        generate_task_module.generate_clip_task(
            {
                "jobId": "job-1",
                "clipId": "clip-1",
                "userId": "user-1",
                "generationCredits": 3,
                "subscriptionTier": "basic",
                "sourceProfile": "source_h1080",
            }
        )

    transcript_updates = [
        payload["transcript"]
        for payload in fake_supabase.clip_updates
        if isinstance(payload, dict) and "transcript" in payload
    ]
    assert len(transcript_updates) >= 1
    saved_transcript = transcript_updates[0]
    assert saved_transcript["metadata"]["batch_split_deferred_render"]["keep_intervals"] == [
        [70.0, 100.0],
        [110.0, 140.0],
    ]
    assert saved_transcript["metadata"]["batch_split_deferred_render"]["transcript_offset_seconds"] == 0.0

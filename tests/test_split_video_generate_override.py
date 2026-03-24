from __future__ import annotations

from types import SimpleNamespace

import pytest

import tasks.clips.generate as generate_task_module


class _FakeResponse:
    def __init__(self, data):
        self.data = data
        self.error = None


class _FakeClipsQuery:
    def __init__(self, clip_row: dict):
        self.clip_row = clip_row
        self._mode = "select"

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        return self

    def update(self, *_args, **_kwargs):
        self._mode = "update"
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        if self._mode == "select":
            return _FakeResponse(self.clip_row)
        return _FakeResponse([self.clip_row])


class _FakeSupabase:
    def __init__(self, clip_row: dict):
        self.clip_row = clip_row

    def table(self, name: str):
        if name != "clips":
            raise AssertionError(f"Unexpected table access: {name}")
        return _FakeClipsQuery(self.clip_row)


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

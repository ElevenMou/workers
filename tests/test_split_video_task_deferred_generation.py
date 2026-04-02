from __future__ import annotations

import sys
import types
from types import SimpleNamespace

openai_stub = types.ModuleType("openai")
openai_stub.NotFoundError = type("NotFoundError", (Exception,), {})
openai_stub.LengthFinishReasonError = type("LengthFinishReasonError", (Exception,), {})


class _OpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


openai_stub.OpenAI = _OpenAI
sys.modules.setdefault("openai", openai_stub)

import tasks.videos.split_video as split_video_module


class _FakeResponse:
    def __init__(self, data=None):
        self.data = data
        self.error = None


class _FakeTable:
    def __init__(self, store: "_FakeSupabase", name: str):
        self.store = store
        self.name = name
        self._mode = "select"
        self._payload = None
        self._filters: dict[str, object] = {}

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        self._filters = {}
        return self

    def update(self, payload):
        self._mode = "update"
        self._payload = payload
        return self

    def insert(self, payload):
        self._mode = "insert"
        self._payload = payload
        return self

    def eq(self, key, value):
        self._filters[str(key)] = value
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.name == "videos" and self._mode == "update":
            self.store.video_updates.append(dict(self._payload or {}))
            return _FakeResponse(data=[dict(self._payload or {})])
        if self.name == "clips" and self._mode == "insert":
            payload = self._payload or []
            rows = list(payload if isinstance(payload, list) else [payload])
            self.store.clip_inserts.append(rows)
            return _FakeResponse(data=rows)
        return _FakeResponse(data=[])


class _FakeSupabase:
    def __init__(self):
        self.video_updates: list[dict] = []
        self.clip_inserts: list[list[dict]] = []

    def table(self, name: str):
        return _FakeTable(self, name)


def test_split_video_task_creates_pending_clip_definitions_without_enqueuing_generation(
    monkeypatch,
    tmp_path,
):
    fake_supabase = _FakeSupabase()
    job_status_updates: list[dict] = []
    video_status_updates: list[tuple[str, dict]] = []

    monkeypatch.setattr(split_video_module, "supabase", fake_supabase)
    monkeypatch.setattr(split_video_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        split_video_module,
        "_load_video_context_row",
        lambda _video_id: {
            "id": "video-1",
            "status": "pending",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Source title",
            "duration_seconds": 125,
            "thumbnail_url": "https://image.test/thumb.jpg",
            "platform": "youtube",
            "external_id": "abc123",
            "raw_video_path": None,
            "raw_video_storage_path": None,
            "transcript": None,
        },
    )
    monkeypatch.setattr(split_video_module, "_is_latest_split_job_for_video", lambda **_kwargs: True)
    monkeypatch.setattr(split_video_module, "create_work_dir", lambda _name: str(tmp_path))
    monkeypatch.setattr(
        split_video_module,
        "VideoDownloader",
        lambda work_dir: SimpleNamespace(work_dir=work_dir),
    )
    monkeypatch.setattr(
        split_video_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(
            find_removable_ranges=lambda *_a, **_k: [],
            generate_segment_titles=lambda segments, **_kwargs: [
                f"Generated title {index + 1}" for index, _segment in enumerate(segments)
            ],
        ),
    )
    monkeypatch.setattr(
        split_video_module,
        "resolve_source_video",
        lambda **_kwargs: SimpleNamespace(
            video_path=str(tmp_path / "source.mkv"),
            storage_path="raw-videos/raw/video-1__source_best_2160.mkv",
            download_metadata={
                "duration": 125,
                "title": "Source title",
                "thumbnail": "https://image.test/thumb.jpg",
                "platform": "youtube",
                "external_id": "abc123",
            },
        ),
    )
    monkeypatch.setattr(
        split_video_module,
        "resolve_source_transcript",
        lambda **_kwargs: SimpleNamespace(
            transcript={
                "source": "youtube",
                "segments": [
                    {"start": 0.0, "end": 40.0, "text": "First section"},
                    {"start": 40.0, "end": 80.0, "text": "Second section"},
                    {"start": 80.0, "end": 125.0, "text": "Third section"},
                ],
            },
            fallback_reason=None,
        ),
    )
    monkeypatch.setattr(
        split_video_module,
        "update_job_status",
        lambda job_id, status, progress, error_message=None, result_data=None, **_kwargs: job_status_updates.append(
            {
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "error_message": error_message,
                "result_data": result_data,
            }
        ),
    )
    monkeypatch.setattr(
        split_video_module,
        "update_video_status",
        lambda video_id, status, **kwargs: video_status_updates.append((status, {"video_id": video_id, **kwargs})),
    )

    split_video_module.split_video_task(
        {
            "jobId": "job-1",
            "videoId": "video-1",
            "userId": "user-1",
            "segmentLengthSeconds": 60,
            "expectedPartCount": 3,
            "expectedGenerationCredits": 9,
            "subscriptionTier": "basic",
            "sourceDetectedLanguage": "en",
        }
    )

    assert len(fake_supabase.clip_inserts) == 1
    inserted_rows = fake_supabase.clip_inserts[0]
    assert len(inserted_rows) == 3
    assert all(row["origin"] == "batch_split" for row in inserted_rows)
    assert all(row["status"] == "pending" for row in inserted_rows)

    final_update = next(
        update
        for update in reversed(job_status_updates)
        if update["status"] == "completed"
    )
    assert final_update["result_data"]["generation_mode"] == "deferred"
    assert len(final_update["result_data"]["created_clip_ids"]) == 3
    assert "child_job_ids" not in final_update["result_data"]
    assert "failed_child_job_ids" not in final_update["result_data"]

    assert video_status_updates[0][0] == "downloading"
    assert video_status_updates[-1][0] == "completed"

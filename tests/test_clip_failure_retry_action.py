from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

if "whisper" not in sys.modules:
    whisper_stub = ModuleType("whisper")
    whisper_stub.load_model = lambda *_args, **_kwargs: SimpleNamespace(
        transcribe=lambda *_a, **_k: {"text": "", "segments": [], "language": "en"}
    )
    sys.modules["whisper"] = whisper_stub

from tasks.clips import custom as custom_task_module
from tasks.clips import generate as generate_task_module
from utils import supabase_client


class _FakeResponse:
    def __init__(self, error=None):
        self.error = error
        self.data = None


class _FakeTableQuery:
    def __init__(self, store: list[dict[str, object]], table_name: str):
        self._store = store
        self._table_name = table_name
        self._payload: dict[str, object] | None = None
        self._filters: dict[str, object] = {}

    def update(self, payload: dict[str, object]):
        self._payload = dict(payload)
        return self

    def eq(self, key: str, value: object):
        self._filters[key] = value
        return self

    def execute(self):
        call = {
            "table": self._table_name,
            "payload": dict(self._payload or {}),
            "filters": dict(self._filters),
        }
        self._store.append(call)
        if self._table_name == "jobs" and "action" in call["payload"] and call["payload"]["action"]:
            # Simulate environments where the jobs.action column has not been deployed yet.
            return _FakeResponse(
                error=SimpleNamespace(message='column "action" of relation "jobs" does not exist')
            )
        return _FakeResponse(error=None)


class _FakeSupabase:
    def __init__(self, store: list[dict[str, object]]):
        self._store = store

    def table(self, table_name: str):
        return _FakeTableQuery(self._store, table_name)


def test_update_job_status_falls_back_without_action_column(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(supabase_client, "supabase", _FakeSupabase(calls))

    supabase_client.update_job_status(
        job_id="job-1",
        status="failed",
        progress=0,
        error="boom",
        action="retry",
    )

    assert calls[0]["table"] == "jobs"
    assert calls[0]["payload"]["action"] == "retry"
    assert calls[1]["table"] == "jobs"
    assert "action" not in calls[1]["payload"]


def test_generate_mark_failed_sets_retry_action(monkeypatch):
    calls: list[dict[str, object]] = []
    job_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(generate_task_module, "supabase", _FakeSupabase(calls))
    monkeypatch.setattr(generate_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        generate_task_module,
        "update_job_status",
        lambda *args, **kwargs: job_updates.append((args, kwargs)),
    )

    generate_task_module._best_effort_mark_failed(
        job_id="job-generate",
        clip_id="clip-1",
        error_msg="failed stage",
    )

    assert job_updates[0][0][:4] == ("job-generate", "failed", 0, "failed stage")
    assert job_updates[0][1].get("action") == "retry"
    clip_updates = [call for call in calls if call["table"] == "clips"]
    assert clip_updates
    assert clip_updates[0]["payload"]["status"] == "failed"
    assert clip_updates[0]["payload"]["action"] == "retry"


def test_custom_mark_failed_sets_retry_action(monkeypatch):
    calls: list[dict[str, object]] = []
    job_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []
    video_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(custom_task_module, "supabase", _FakeSupabase(calls))
    monkeypatch.setattr(custom_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        custom_task_module,
        "update_job_status",
        lambda *args, **kwargs: job_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        custom_task_module,
        "update_video_status",
        lambda *args, **kwargs: video_updates.append((args, kwargs)),
    )

    custom_task_module._best_effort_mark_failed(
        job_id="job-custom",
        clip_id="clip-2",
        video_id="video-2",
        error_msg="custom failed stage",
    )

    assert job_updates[0][0][:4] == ("job-custom", "failed", 0, "custom failed stage")
    assert job_updates[0][1].get("action") == "retry"
    clip_updates = [call for call in calls if call["table"] == "clips"]
    assert clip_updates
    assert clip_updates[0]["payload"]["status"] == "failed"
    assert clip_updates[0]["payload"]["action"] == "retry"
    assert video_updates[0][0][:2] == ("video-2", "failed")
    assert video_updates[0][1].get("error_message") == "custom failed stage"

from __future__ import annotations

from worker_supervisor import runtime
from worker_supervisor import startup


class _FakePipeline:
    def __init__(self):
        self.commands: list[tuple] = []

    def delete(self, key):
        self.commands.append(("delete", key))
        return self

    def srem(self, key, value):
        self.commands.append(("srem", key, value))
        return self

    def execute(self):
        return []


class _FakeConn:
    def __init__(self):
        self.pipeline_obj = _FakePipeline()

    def smembers(self, key):
        if key == "rq:queues":
            return {b"rq:queue:clip-generation", b"rq:queue:video-processing"}
        return set()

    def pipeline(self):
        return self.pipeline_obj


def test_cleanup_named_workers_force_removes_registration(monkeypatch):
    conn = _FakeConn()

    monkeypatch.setattr(runtime.Worker, "all", lambda connection: [])

    runtime.cleanup_named_workers(conn, {"clip-worker-3"})

    expected_key = "rq:worker:clip-worker-3"
    commands = conn.pipeline_obj.commands
    assert ("delete", expected_key) in commands
    assert ("srem", "rq:workers", expected_key) in commands
    assert ("srem", "rq:workers:clip-generation", expected_key) in commands
    assert ("srem", "rq:workers:video-processing", expected_key) in commands


def test_run_worker_retries_once_after_name_collision(monkeypatch):
    fake_conn = object()
    work_calls: list[str] = []
    cleanup_calls: list[str] = []

    class _FakeWorker:
        def __init__(self, queues, connection, name):
            self.name = name

        def work(self, with_scheduler=False):
            work_calls.append(self.name)
            if len(work_calls) == 1:
                raise ValueError("There exists an active worker named 'clip-worker-3' already")

    import utils.redis_client as redis_client

    monkeypatch.setattr(redis_client, "get_redis_connection", lambda: fake_conn)
    monkeypatch.setattr(redis_client, "get_queues", lambda names, conn: ["q"])
    monkeypatch.setattr(runtime, "worker_cls", lambda: _FakeWorker)
    monkeypatch.setattr(
        runtime,
        "_force_remove_worker_registration",
        lambda conn, worker_name: cleanup_calls.append(worker_name),
    )

    runtime.run_worker(["clip-generation"], "clip-worker-3")

    assert cleanup_calls == ["clip-worker-3"]
    assert work_calls == ["clip-worker-3", "clip-worker-3"]


class _FakeSupabaseResponse:
    def __init__(self, data):
        self.data = data
        self.error = None


class _FakeSupabaseTable:
    def __init__(self, owner, table_name: str):
        self.owner = owner
        self.table_name = table_name
        self.mode = "select"
        self.update_payload = {}
        self.filters: dict[str, object] = {}

    def select(self, *_args, **_kwargs):
        self.mode = "select"
        return self

    def update(self, payload: dict):
        self.mode = "update"
        self.update_payload = dict(payload)
        return self

    def eq(self, column: str, value):
        self.filters[column] = value
        return self

    def lt(self, column: str, value):
        self.filters[column] = ("lt", value)
        return self

    def is_(self, column: str, value):
        self.filters[column] = ("is", value)
        return self

    def limit(self, _count: int):
        return self

    def execute(self):
        if self.table_name == "jobs" and self.mode == "select":
            return _FakeSupabaseResponse(self.owner.jobs_rows)

        if self.mode == "update":
            self.owner.updates.append(
                (self.table_name, dict(self.update_payload), dict(self.filters))
            )
            return _FakeSupabaseResponse({})

        return _FakeSupabaseResponse({})


class _FakeSupabase:
    def __init__(self, jobs_rows: list[dict]):
        self.jobs_rows = list(jobs_rows)
        self.updates: list[tuple[str, dict, dict]] = []

    def table(self, table_name: str):
        return _FakeSupabaseTable(self, table_name)


def test_runtime_stale_recovery_skips_jobs_in_started_registry(monkeypatch):
    fake_supabase = _FakeSupabase(
        jobs_rows=[
            {
                "id": "job-stale-1",
                "type": "generate_clip",
                "clip_id": "clip-1",
                "video_id": "video-1",
                "started_at": "2026-02-25T00:00:00+00:00",
                "created_at": "2026-02-25T00:00:00+00:00",
            },
            {
                "id": "job-active-2",
                "type": "analyze_video",
                "clip_id": None,
                "video_id": "video-2",
                "started_at": "2026-02-25T00:00:00+00:00",
                "created_at": "2026-02-25T00:00:00+00:00",
            },
        ]
    )
    failed_jobs: list[str] = []

    import utils.supabase_client as supabase_client

    monkeypatch.setattr(supabase_client, "supabase", fake_supabase)
    monkeypatch.setattr(
        startup,
        "collect_started_job_ids",
        lambda conn, queue_names: {"job-active-2"},
    )
    monkeypatch.setattr(
        startup,
        "mark_jobs_failed",
        lambda job_ids, reason: failed_jobs.extend(job_ids),
    )

    recovered = startup.recover_stale_processing_rows(
        conn=object(),
        queue_names=["clip-generation", "video-processing"],
        stale_seconds=120,
    )

    assert recovered == 1
    assert failed_jobs == ["job-stale-1"]
    assert any(
        table == "clips" and filters.get("id") == "clip-1"
        for table, _payload, filters in fake_supabase.updates
    )
    assert not any(
        table == "videos" and filters.get("id") == "video-2"
        for table, _payload, filters in fake_supabase.updates
    )


def test_runtime_stale_recovery_marks_analyze_video_rows_failed(monkeypatch):
    fake_supabase = _FakeSupabase(
        jobs_rows=[
            {
                "id": "job-analyze-1",
                "type": "analyze_video",
                "clip_id": None,
                "video_id": "video-7",
                "started_at": "2026-02-25T00:00:00+00:00",
                "created_at": "2026-02-25T00:00:00+00:00",
            }
        ]
    )
    failed_jobs: list[str] = []

    import utils.supabase_client as supabase_client

    monkeypatch.setattr(supabase_client, "supabase", fake_supabase)
    monkeypatch.setattr(startup, "collect_started_job_ids", lambda conn, queue_names: set())
    monkeypatch.setattr(
        startup,
        "mark_jobs_failed",
        lambda job_ids, reason: failed_jobs.extend(job_ids),
    )

    recovered = startup.recover_stale_processing_rows(
        conn=object(),
        queue_names=["video-processing"],
        stale_seconds=120,
    )

    assert recovered == 1
    assert failed_jobs == ["job-analyze-1"]
    assert any(
        table == "videos" and filters.get("id") == "video-7"
        for table, _payload, filters in fake_supabase.updates
    )

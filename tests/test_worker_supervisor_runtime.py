from __future__ import annotations

from worker_supervisor import runtime


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

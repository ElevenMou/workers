from __future__ import annotations

from types import SimpleNamespace

from worker_supervisor import supervisor


def test_run_supervisor_maintenance_role_delegates_to_maintenance_loop(monkeypatch):
    import config
    import utils.redis_client as redis_client

    monkeypatch.setattr(config, "SUPERVISOR_ROLE", "maintenance")
    monkeypatch.setattr(config, "WORKER_INSTANCE_ID", "maint-a")
    monkeypatch.setattr(config, "NUM_VIDEO_WORKERS", 0)
    monkeypatch.setattr(config, "NUM_CLIP_WORKERS", 0)
    monkeypatch.setattr(config, "RAW_VIDEO_CLEANUP_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(config, "CLIP_ASSET_CLEANUP_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(config, "PROCESSING_JOB_STALE_SECONDS", 120)
    monkeypatch.setattr(config, "MAINTENANCE_LEADER_LOCK_TTL_SECONDS", 15)
    monkeypatch.setattr(config, "MAINTENANCE_LEADER_RENEW_SECONDS", 5)
    monkeypatch.setattr(config, "validate_env", lambda: None)
    monkeypatch.setattr(redis_client, "get_redis_connection", lambda: object())

    called = {"value": False}

    def _fake_loop(**_kwargs):
        called["value"] = True
        return 0

    monkeypatch.setattr(supervisor, "_run_maintenance_loop", _fake_loop)

    assert supervisor.run_supervisor() == 0
    assert called["value"] is True


def test_worker_role_uses_instance_scoped_worker_names(monkeypatch):
    import config
    import utils.redis_client as redis_client

    class _FakeProcess:
        pid = 4321

        def is_alive(self):
            return True

    cleanup_calls: list[set[str]] = []
    stop_calls: list[str] = []
    spawn_calls: list[str] = []

    monkeypatch.setattr(config, "SUPERVISOR_ROLE", "worker")
    monkeypatch.setattr(config, "WORKER_INSTANCE_ID", "node-A")
    monkeypatch.setattr(config, "NUM_VIDEO_WORKERS", 1)
    monkeypatch.setattr(config, "NUM_CLIP_WORKERS", 0)
    monkeypatch.setattr(config, "RAW_VIDEO_CLEANUP_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(config, "CLIP_ASSET_CLEANUP_INTERVAL_SECONDS", 10)
    monkeypatch.setattr(config, "PROCESSING_JOB_STALE_SECONDS", 120)
    monkeypatch.setattr(config, "validate_env", lambda: None)
    monkeypatch.setenv("WORKER_MODE", "video")

    fake_conn = SimpleNamespace(get=lambda *_a, **_k: None)
    monkeypatch.setattr(redis_client, "get_redis_connection", lambda: fake_conn)
    monkeypatch.setattr(
        redis_client,
        "set_group_worker_scale_target",
        lambda **kwargs: int(kwargs["workers"]),
    )
    monkeypatch.setattr(
        redis_client,
        "get_group_worker_scale_target",
        lambda **kwargs: int(kwargs["default"]),
    )

    monkeypatch.setattr(
        supervisor,
        "cleanup_named_workers",
        lambda _conn, names: cleanup_calls.append(set(names)),
    )
    monkeypatch.setattr(
        supervisor,
        "spawn_worker",
        lambda _queue_names, worker_name: (spawn_calls.append(worker_name) or _FakeProcess()),
    )
    monkeypatch.setattr(
        supervisor,
        "stop_worker_process",
        lambda _conn, name, _proc: stop_calls.append(name),
    )
    sleep_calls = {"count": 0}

    def _sleep_and_interrupt(_seconds: float):
        sleep_calls["count"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(supervisor.time, "sleep", _sleep_and_interrupt)

    assert supervisor.run_supervisor() == 0
    assert spawn_calls
    assert all("node-A" in worker_name for worker_name in spawn_calls)
    assert stop_calls
    assert any("node-A" in next(iter(call)) for call in cleanup_calls if call)

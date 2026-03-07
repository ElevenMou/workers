from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api_app.auth as auth_module
import api_app.helpers as api_helpers
from services.clips.quality_policy import (
    resolve_clip_quality_policy,
    resolve_effective_output_quality,
)
from tasks.clips.helpers import lifecycle as lifecycle_helpers
from utils import redis_client


def _build_request(*, authorization: str | None = None, client_host: str = "203.0.113.10") -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if authorization:
        headers.append((b"authorization", authorization.encode("utf-8")))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers,
        "client": (client_host, 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_get_user_rate_key_uses_local_extractor(monkeypatch):
    monkeypatch.setattr(auth_module, "extract_rate_limit_user_id", lambda _token: "user-123")
    monkeypatch.setattr(
        auth_module,
        "_fetch_user_from_token",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    request = _build_request(authorization="Bearer test-token")
    assert auth_module.get_user_rate_key(request) == "user:user-123"


def test_get_user_rate_key_falls_back_to_ip_on_invalid_token(monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "extract_rate_limit_user_id",
        lambda _token: (_ for _ in ()).throw(RuntimeError("invalid token")),
    )
    request = _build_request(authorization="Bearer bad-token", client_host="198.51.100.8")
    assert auth_module.get_user_rate_key(request) == "198.51.100.8"


class _FakeDeleteQuery:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.eq_calls: list[tuple[str, object]] = []
        self.executed = False

    def eq(self, key: str, value: object):
        self.eq_calls.append((key, value))
        return self

    def execute(self):
        self.executed = True
        return SimpleNamespace(data=[])


class _FakeTable:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.delete_query = _FakeDeleteQuery(table_name)

    def delete(self):
        return self.delete_query


class _FakeSupabase:
    def __init__(self):
        self.tables: dict[str, _FakeTable] = {}

    def table(self, table_name: str):
        table = self.tables.get(table_name)
        if table is None:
            table = _FakeTable(table_name)
            self.tables[table_name] = table
        return table


def test_enqueue_or_fail_returns_429_and_runs_cleanup(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(api_helpers, "supabase", fake_supabase)
    monkeypatch.setattr(
        api_helpers,
        "enqueue_job",
        lambda *_a, **_k: (_ for _ in ()).throw(redis_client.QueueFullError("full")),
    )

    cleanup_called = {"value": False}

    def _cleanup():
        cleanup_called["value"] = True

    with pytest.raises(HTTPException) as exc:
        api_helpers.enqueue_or_fail(
            queue_name="clip-generation",
            task_path="tasks.fake",
            job_data={"id": 1},
            job_id="job-1",
            user_id="user-1",
            job_type="generate_clip",
            video_id="video-1",
            clip_id="clip-1",
            on_queue_full_cleanup=_cleanup,
        )

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "30"
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail.get("code") == "queue_full"
    assert cleanup_called["value"] is True

    jobs_table = fake_supabase.tables["jobs"]
    assert jobs_table.delete_query.executed is True
    assert ("id", "job-1") in jobs_table.delete_query.eq_calls


@dataclass
class _FakeRedis:
    strings: dict[str, object]
    hashes: dict[str, dict]

    def __init__(self):
        self.strings = {}
        self.hashes = {}

    def get(self, key: str):
        return self.strings.get(key)

    def set(self, key: str, value: object, nx: bool = False, ex: int | None = None):
        del ex
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        return True

    def delete(self, key: str):
        self.strings.pop(key, None)
        return 1

    def eval(self, script: str, _numkeys: int, *params):
        key = str(params[0])
        if "max_depth" in script:
            max_depth = int(params[1])
            current = int(self.strings.get(key, 0))
            if current >= max_depth:
                return -1
            current += 1
            if current > max_depth:
                return -1
            self.strings[key] = current
            return current
        current = int(self.strings.get(key, 0))
        if current > 0:
            current -= 1
            self.strings[key] = current
        return current

    def hgetall(self, key: str):
        return self.hashes.get(key, {})

    def hset(self, key: str, mapping: dict):
        bucket = self.hashes.setdefault(key, {})
        bucket.update(mapping)
        return 1

    def incr(self, key: str):
        current = int(self.strings.get(key, 0))
        current += 1
        self.strings[key] = current
        return current

    def llen(self, _key: str):
        return 0


def test_worker_scale_legacy_hash_fallback_reads_values():
    fake_conn = _FakeRedis()
    fake_conn.hashes[redis_client.WORKER_SCALE_KEY] = {
        "video_workers": "3",
        "clip_workers": "5",
    }

    video_workers, clip_workers, social_workers = redis_client.get_worker_scale_target(
        connection=fake_conn,
        default_video=1,
        default_clip=1,
        default_social=2,
    )
    assert video_workers == 3
    assert clip_workers == 5
    assert social_workers == 2


def test_set_group_worker_scale_target_does_not_mutate_other_group():
    fake_conn = _FakeRedis()
    redis_client.set_group_worker_scale_target(
        group="video",
        workers=4,
        connection=fake_conn,
        default=1,
    )
    redis_client.set_group_worker_scale_target(
        group="clip",
        workers=7,
        connection=fake_conn,
        default=1,
    )

    redis_client.set_group_worker_scale_target(
        group="video",
        workers=9,
        connection=fake_conn,
        default=1,
    )

    assert int(fake_conn.get(redis_client.WORKER_SCALE_VIDEO_KEY)) == 9
    assert int(fake_conn.get(redis_client.WORKER_SCALE_CLIP_KEY)) == 7


def test_release_job_admission_is_idempotent():
    fake_conn = _FakeRedis()
    fake_conn.strings[redis_client.ADMISSION_VIDEO_KEY] = 1
    fake_conn.strings[f"{redis_client.ADMISSION_JOB_GROUP_PREFIX}job-1"] = "video"

    assert redis_client.release_job_admission("job-1", connection=fake_conn) is True
    assert int(fake_conn.get(redis_client.ADMISSION_VIDEO_KEY)) == 0
    assert redis_client.release_job_admission("job-1", connection=fake_conn) is False


def test_enqueue_job_sets_retry_policy_and_admission_meta(monkeypatch):
    fake_conn = _FakeRedis()
    captured: dict[str, object] = {}

    class _FakeQueue:
        def enqueue(self, task_path, job_data, **kwargs):
            captured["task_path"] = task_path
            captured["job_data"] = job_data
            captured["kwargs"] = kwargs
            return SimpleNamespace(id=kwargs.get("job_id"))

    monkeypatch.setattr(redis_client, "get_redis_connection", lambda: fake_conn)
    monkeypatch.setattr(redis_client, "get_queue", lambda _name, _conn: _FakeQueue())

    result = redis_client.enqueue_job(
        "video-processing",
        "tasks.fake.task",
        {"x": 1},
        job_id="job-99",
    )

    assert result.id == "job-99"
    kwargs = captured["kwargs"]
    assert kwargs["job_id"] == "job-99"
    assert kwargs["meta"]["admission_group"] == "video"
    assert kwargs["meta"]["admission_token"] == 1
    assert kwargs["retry"].max == 3


def test_health_metrics_requires_auth(client):
    response = client.get("/health/metrics")
    assert response.status_code == 401


def test_quality_policy_premium_short_clip_promotes_quality():
    policy = resolve_clip_quality_policy(
        tier="pro",
        clip_duration_seconds=45.0,
        requested_output_quality="medium",
    )
    assert policy["source_max_height"] == 2160
    assert policy["output_quality"] == "high"
    assert policy["profile"] == "premium_short_clip"


def test_quality_policy_basic_keeps_balanced_1080():
    policy = resolve_clip_quality_policy(
        tier="basic",
        clip_duration_seconds=45.0,
        requested_output_quality="medium",
    )
    assert policy["source_max_height"] == 1080
    assert policy["output_quality"] == "medium"
    assert policy["profile"] == "balanced_1080"


def test_quality_override_never_downgrades_template_quality():
    assert (
        resolve_effective_output_quality(
            template_quality="high",
            policy_override_quality="medium",
        )
        == "high"
    )
    assert (
        resolve_effective_output_quality(
            template_quality="high",
            policy_override_quality="low",
        )
        == "high"
    )
    assert (
        resolve_effective_output_quality(
            template_quality="medium",
            policy_override_quality="high",
        )
        == "high"
    )


def test_upload_clip_disallow_reencode_fails_before_preupload_optimization(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    reencode_called = {"value": False}

    monkeypatch.setattr(
        lifecycle_helpers,
        "_get_generated_clips_bucket_limit_bytes",
        lambda **_kwargs: 32,
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_upload_with_duplicate_replace",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("upload should not be called")),
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_reencode_clip_for_upload",
        lambda **_kwargs: reencode_called.__setitem__("value", True),
    )

    with pytest.raises(RuntimeError, match="quality-preserving mode disables upload re-encoding"):
        lifecycle_helpers.upload_clip_with_replace(
            local_clip_path=str(clip_path),
            storage_path="clips/test.mp4",
            job_id="job-1",
            logger=SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None),
            allow_reencode=False,
        )

    assert reencode_called["value"] is False


def test_upload_clip_disallow_reencode_fails_on_payload_too_large_without_retry(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    reencode_called = {"value": False}

    monkeypatch.setattr(
        lifecycle_helpers,
        "_get_generated_clips_bucket_limit_bytes",
        lambda **_kwargs: None,
    )

    def _raise_payload_too_large(**_kwargs):
        raise Exception(
            {
                "statusCode": 400,
                "error": "payload too large",
                "message": "maximum allowed size exceeded",
            }
        )

    monkeypatch.setattr(
        lifecycle_helpers,
        "_upload_with_duplicate_replace",
        _raise_payload_too_large,
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_reencode_clip_for_upload",
        lambda **_kwargs: reencode_called.__setitem__("value", True),
    )

    with pytest.raises(
        RuntimeError,
        match="quality-preserving mode disables fallback re-encoding",
    ):
        lifecycle_helpers.upload_clip_with_replace(
            local_clip_path=str(clip_path),
            storage_path="clips/test.mp4",
            job_id="job-2",
            logger=SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None),
            allow_reencode=False,
        )

    assert reencode_called["value"] is False

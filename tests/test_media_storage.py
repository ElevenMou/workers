from __future__ import annotations

import os
import threading
from urllib.parse import parse_qs, urlparse

import api_app.routers.media as media_router
import utils.media_storage as media_storage
import utils.minio_client as minio_client
import pytest


class _FakeRedis:
    def __init__(self):
        self._values: dict[str, str] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        del ex
        with self._lock:
            if nx and key in self._values:
                return False
            self._values[key] = value
            return True

    def get(self, key: str):
        with self._lock:
            return self._values.get(key)

    def delete(self, key: str):
        with self._lock:
            self._values.pop(key, None)
        return 1


def test_preferred_source_video_order_is_minio_first():
    assert media_storage.preferred_source_video_order() == ("storage", "local")


def test_signed_worker_clip_url_round_trips(monkeypatch):
    monkeypatch.setattr(media_storage, "WORKER_PUBLIC_BASE_URL", "http://testserver")
    monkeypatch.setattr(media_storage, "WORKER_MEDIA_SIGNING_SECRET", "test-media-secret")

    signed_url = media_storage.build_worker_clip_url("clip-123", expires_in_seconds=60)

    assert signed_url is not None
    parsed = urlparse(signed_url)
    query = parse_qs(parsed.query)

    assert media_storage.verify_signed_worker_media_request(
        relative_path=parsed.path,
        expires=int(query["expires"][0]),
        signature=query["sig"][0],
    )
    assert not media_storage.verify_signed_worker_media_request(
        relative_path=parsed.path,
        expires=int(query["expires"][0]),
        signature="bad-signature",
    )


def test_signed_worker_clip_url_uses_caddy_domain_when_public_base_url_missing(monkeypatch):
    monkeypatch.setattr(media_storage, "WORKER_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(media_storage, "WORKER_MEDIA_SIGNING_SECRET", "test-media-secret")
    monkeypatch.setenv("CADDY_DOMAIN", "api.clipscut.pro")

    signed_url = media_storage.build_worker_clip_url("clip-123", expires_in_seconds=60)

    assert signed_url is not None
    parsed = urlparse(signed_url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "api.clipscut.pro"
    assert media_storage.verify_signed_worker_media_request(
        relative_path=parsed.path,
        expires=int(query["expires"][0]),
        signature=query["sig"][0],
    )


def test_signed_worker_clip_url_ignores_local_worker_base_url_in_production(monkeypatch):
    monkeypatch.setattr(media_storage, "IS_PRODUCTION", True)
    monkeypatch.setattr(media_storage, "WORKER_PUBLIC_BASE_URL", "http://localhost:8080")
    monkeypatch.setattr(media_storage, "WORKER_MEDIA_SIGNING_SECRET", "test-media-secret")
    monkeypatch.setenv("CADDY_DOMAIN", "api.clipscut.pro")

    signed_url = media_storage.build_worker_clip_url("clip-123", expires_in_seconds=60)

    assert signed_url is not None
    parsed = urlparse(signed_url)
    assert parsed.netloc == "api.clipscut.pro"


def test_signed_worker_clip_url_uses_internal_token_when_media_secret_missing(monkeypatch):
    monkeypatch.setattr(media_storage, "WORKER_PUBLIC_BASE_URL", "https://api.clipscut.pro")
    monkeypatch.setattr(media_storage, "WORKER_MEDIA_SIGNING_SECRET", "")
    monkeypatch.setattr(media_storage, "WORKER_INTERNAL_API_TOKEN", "internal-token")

    signed_url = media_storage.build_worker_clip_url("clip-123", expires_in_seconds=60)

    assert signed_url is not None
    parsed = urlparse(signed_url)
    query = parse_qs(parsed.query)

    assert media_storage.verify_signed_worker_media_request(
        relative_path=parsed.path,
        expires=int(query["expires"][0]),
        signature=query["sig"][0],
    )


def test_resolve_generated_clip_path_prefers_existing_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))

    local_path = media_storage.resolve_generated_clip_local_path("clips/local-first.mp4")
    media_storage._write_bytes_atomically(local_path, b"local-bytes")

    class _UnexpectedMinioClient:
        def fget_object(self, *_args, **_kwargs):
            raise AssertionError("remote download should not be used when local file exists")

    monkeypatch.setattr(
        minio_client,
        "get_minio_client",
        lambda: _UnexpectedMinioClient(),
    )

    resolved_path = media_storage.resolve_generated_clip_path("clips/local-first.mp4")

    assert resolved_path == local_path


def test_resolve_generated_clip_path_materializes_minio_object(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))
    download_calls: list[tuple[str, str, str]] = []

    class _FakeMinioClient:
        def fget_object(self, bucket, object_name, file_path):
            download_calls.append((bucket, object_name, file_path))
            with open(file_path, "wb") as handle:
                handle.write(b"remote-bytes")

    monkeypatch.setattr(
        minio_client,
        "get_minio_client",
        lambda: _FakeMinioClient(),
    )

    resolved_path = media_storage.resolve_generated_clip_path(
        "generated-clips/clips/remote.mp4"
    )

    assert resolved_path is not None
    assert download_calls == [
        ("generated-clips", "clips/remote.mp4", resolved_path)
    ]
    with open(resolved_path, "rb") as handle:
        assert handle.read() == b"remote-bytes"


def test_resolve_generated_clip_path_raises_missing_object_error(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))

    class _MissingObjectClient:
        def fget_object(self, _bucket, _object_name, _file_path):
            error = RuntimeError("missing")
            setattr(error, "code", "NoSuchKey")
            raise error

    monkeypatch.setattr(
        minio_client,
        "get_minio_client",
        lambda: _MissingObjectClient(),
    )

    with pytest.raises(media_storage.GeneratedClipStorageError) as exc:
        media_storage.resolve_generated_clip_path("clips/remote.mp4", raise_on_error=True)

    assert exc.value.reason == "missing_object"
    assert exc.value.object_name == "clips/remote.mp4"
    assert exc.value.recoverable is False


def test_resolve_generated_clip_path_deduplicates_concurrent_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))
    fake_redis = _FakeRedis()
    monkeypatch.setattr(media_storage, "get_redis_connection", lambda: fake_redis)
    monkeypatch.setattr(media_storage, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 5)
    monkeypatch.setattr(media_storage, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)

    download_started = threading.Event()
    release_download = threading.Event()
    download_calls: list[tuple[str, str, str, str | None]] = []

    class _FakeMinioClient:
        def fget_object(self, bucket, object_name, file_path, *, tmp_file_path=None):
            download_calls.append((bucket, object_name, file_path, tmp_file_path))
            assert tmp_file_path is not None
            with open(tmp_file_path, "wb") as handle:
                handle.write(b"remote-bytes")
            download_started.set()
            assert release_download.wait(timeout=1)
            os.replace(tmp_file_path, file_path)

    monkeypatch.setattr(
        minio_client,
        "get_minio_client",
        lambda: _FakeMinioClient(),
    )

    results: dict[str, str] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _run(worker_name: str):
        try:
            barrier.wait(timeout=2)
            results[worker_name] = media_storage.resolve_generated_clip_path(
                "clips/remote.mp4",
                raise_on_error=True,
            )
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    first = threading.Thread(target=_run, args=("worker-a",), daemon=True)
    second = threading.Thread(target=_run, args=("worker-b",), daemon=True)
    first.start()
    second.start()
    assert download_started.wait(timeout=1)
    release_download.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not errors
    assert len(download_calls) == 1
    assert len(results) == 2
    resolved_paths = set(results.values())
    assert len(resolved_paths) == 1
    resolved_path = next(iter(resolved_paths))
    assert resolved_path == media_storage.resolve_generated_clip_local_path("clips/remote.mp4")
    assert download_calls[0][3] != resolved_path
    with open(resolved_path, "rb") as handle:
        assert handle.read() == b"remote-bytes"


def test_resolve_generated_clip_path_reuses_materialized_file_after_download_exception(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))
    monkeypatch.setattr(media_storage, "get_redis_connection", lambda: None)

    class _LateFailureClient:
        def fget_object(self, _bucket, _object_name, file_path, *, tmp_file_path=None):
            with open(tmp_file_path, "wb") as handle:
                handle.write(b"remote-bytes")
            os.replace(tmp_file_path, file_path)
            raise RuntimeError("late failure after materialization")

    monkeypatch.setattr(
        minio_client,
        "get_minio_client",
        lambda: _LateFailureClient(),
    )

    resolved_path = media_storage.resolve_generated_clip_path(
        "clips/remote.mp4",
        raise_on_error=True,
    )

    assert resolved_path == media_storage.resolve_generated_clip_local_path("clips/remote.mp4")
    with open(resolved_path, "rb") as handle:
        assert handle.read() == b"remote-bytes"


def test_stream_clip_media_supports_range_requests(client, tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "WORKER_PUBLIC_BASE_URL", "http://testserver")
    monkeypatch.setattr(media_storage, "WORKER_MEDIA_SIGNING_SECRET", "test-media-secret")

    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"0123456789")
    monkeypatch.setattr(media_router, "_resolve_clip_media_path", lambda _clip_id: str(clip_path))

    signed_url = media_storage.build_worker_clip_url("clip-123", expires_in_seconds=60)
    assert signed_url is not None
    parsed = urlparse(signed_url)

    response = client.get(
        f"{parsed.path}?{parsed.query}",
        headers={"range": "bytes=2-5"},
    )

    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.content == b"2345"


def test_delete_video_storage_endpoint_removes_local_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(media_router, "WORKER_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setattr(media_storage, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))

    local_raw_path = tmp_path / "downloads" / "video-123.mp4"
    local_raw_path.parent.mkdir(parents=True, exist_ok=True)
    local_raw_path.write_bytes(b"video")

    cached_raw_path = tmp_path / "cache" / "video-123.mp4"
    cached_raw_path.parent.mkdir(parents=True, exist_ok=True)
    cached_raw_path.write_bytes(b"cached")

    response = client.post(
        "/internal/storage/delete-videos",
        headers={"x-worker-internal-token": "internal-token"},
        json={
            "rawVideoPaths": [str(local_raw_path)],
            "rawVideoStoragePaths": ["raw/video-123.mp4"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"requested": 1, "removed": 2}
    assert not local_raw_path.exists()
    assert not cached_raw_path.exists()

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import api_app.routers.media as media_router
import utils.media_storage as media_storage


def test_preferred_source_video_order_changes_with_provider(monkeypatch):
    monkeypatch.setattr(media_storage, "MEDIA_STORAGE_PROVIDER", "local")
    assert media_storage.get_media_storage_provider().name == "local"
    assert media_storage.preferred_source_video_order() == ("local", "storage")

    monkeypatch.setattr(media_storage, "MEDIA_STORAGE_PROVIDER", "supabase")
    assert media_storage.get_media_storage_provider().name == "supabase"
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


def test_resolve_generated_clip_path_prefers_existing_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))

    local_path = media_storage.resolve_generated_clip_local_path("clips/local-first.mp4")
    media_storage._write_bytes_atomically(local_path, b"local-bytes")

    download_calls = []

    class _UnexpectedStorage:
        def download(self, _path):
            download_calls.append(_path)
            raise AssertionError("remote download should not be used when local file exists")

    monkeypatch.setattr(
        media_storage,
        "supabase",
        SimpleNamespace(storage=SimpleNamespace(from_=lambda _bucket: _UnexpectedStorage())),
    )

    resolved_path = media_storage.resolve_generated_clip_path("clips/local-first.mp4")

    assert resolved_path == local_path
    assert download_calls == []


def test_resolve_generated_clip_path_materializes_supabase_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(media_storage, "LOCAL_MEDIA_ROOT", str(tmp_path / "media"))

    download_calls: list[str] = []

    class _FakeStorage:
        def download(self, path):
            download_calls.append(path)
            return b"remote-bytes"

    monkeypatch.setattr(
        media_storage,
        "supabase",
        SimpleNamespace(storage=SimpleNamespace(from_=lambda _bucket: _FakeStorage())),
    )

    resolved_path = media_storage.resolve_generated_clip_path("generated-clips/clips/remote.mp4")

    assert resolved_path is not None
    assert download_calls == ["clips/remote.mp4"]
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

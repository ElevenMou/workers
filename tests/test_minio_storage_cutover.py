from __future__ import annotations

from pathlib import Path

from services.social import media as social_media
from tasks.clips.helpers import source_video
from utils import minio_client


def test_source_video_resolution_downloads_from_minio(tmp_path, monkeypatch):
    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    download_calls: list[tuple[str, str, str]] = []

    class _FakeMinioClient:
        def fget_object(self, bucket, object_name, file_path):
            download_calls.append((bucket, object_name, file_path))
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_bytes(b"video-bytes")

    monkeypatch.setattr(minio_client, "get_minio_client", lambda: _FakeMinioClient())
    monkeypatch.setattr(
        source_video,
        "_resolve_existing_video_path",
        lambda **kwargs: (kwargs["raw_video_path"], 1920, 1080),
    )

    resolved = source_video._resolve_storage_video_path(
        video_id="video-1",
        source_profile="source_h1080",
        job_id="job-1",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        raw_video_storage_path="raw-videos/raw/video-1.mp4",
        force_refresh_cache=True,
    )

    assert resolved == (str(tmp_path / "cache" / "video-1__source_h1080.mp4"), 1920, 1080)
    assert download_calls == [
        (
            "raw-videos",
            "raw/video-1.mp4",
            str(tmp_path / "cache" / "video-1__source_h1080.mp4.tmp"),
        )
    ]


def test_social_media_prefers_public_worker_clip_urls(monkeypatch):
    minio_calls: list[tuple[str, str | None, int]] = []
    worker_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        social_media,
        "build_worker_clip_url",
        lambda clip_id, *, expires_in_seconds=3600: worker_calls.append(
            (clip_id, expires_in_seconds)
        )
        or "https://api.clipscut.pro/media/clips/clip-1?sig=test",
    )
    monkeypatch.setattr(
        social_media,
        "create_minio_signed_clip_url",
        lambda storage_path, *, clip_id=None, expires_in_seconds=3600: minio_calls.append(
            (storage_path, clip_id, expires_in_seconds)
        )
        or "http://localhost:9000/generated-clips/clip-1.mp4",
    )

    signed_url = social_media.create_signed_clip_url(
        "generated-clips/clip-1.mp4",
        expires_in_seconds=90,
    )

    assert signed_url == "https://api.clipscut.pro/media/clips/clip-1?sig=test"
    assert worker_calls == [("clip-1", 90)]
    assert minio_calls == [("generated-clips/clip-1.mp4", "clip-1", 90)]


def test_social_media_falls_back_to_minio_when_worker_url_is_not_public(monkeypatch):
    monkeypatch.setattr(
        social_media,
        "build_worker_clip_url",
        lambda _clip_id, *, expires_in_seconds=3600: f"http://localhost:8001/media?expires={expires_in_seconds}",
    )
    monkeypatch.setattr(
        social_media,
        "create_minio_signed_clip_url",
        lambda _storage_path, *, clip_id=None, expires_in_seconds=3600: (
            f"https://storage.example/{clip_id}?expires={expires_in_seconds}"
        ),
    )

    signed_url = social_media.create_signed_clip_url(
        "generated-clips/clip-1.mp4",
        expires_in_seconds=90,
    )

    assert signed_url == "https://storage.example/clip-1?expires=90"


def test_social_media_returns_none_when_no_public_media_url_exists(monkeypatch):
    monkeypatch.setattr(
        social_media,
        "build_worker_clip_url",
        lambda _clip_id, *, expires_in_seconds=3600: f"http://localhost:8080/media?expires={expires_in_seconds}",
    )
    monkeypatch.setattr(
        social_media,
        "create_minio_signed_clip_url",
        lambda _storage_path, *, clip_id=None, expires_in_seconds=3600: (
            f"http://localhost:9000/generated-clips/{clip_id}?expires={expires_in_seconds}"
        ),
    )

    signed_url = social_media.create_signed_clip_url(
        "generated-clips/clip-1.mp4",
        expires_in_seconds=90,
    )

    assert signed_url is None


def test_initialize_minio_storage_respects_skip_flag_and_runs_once(monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr(
        minio_client,
        "ensure_buckets",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )

    monkeypatch.setattr(minio_client, "_storage_initialized", False)
    monkeypatch.setenv("MINIO_SKIP_STARTUP_READINESS", "true")
    minio_client.initialize_minio_storage()
    assert calls["count"] == 0

    monkeypatch.setattr(minio_client, "_storage_initialized", False)
    monkeypatch.delenv("MINIO_SKIP_STARTUP_READINESS", raising=False)
    minio_client.initialize_minio_storage()
    minio_client.initialize_minio_storage()
    assert calls["count"] == 1

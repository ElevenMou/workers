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


def test_social_media_presigns_clips_from_minio(monkeypatch):
    calls: list[tuple[str, str | None, int]] = []

    monkeypatch.setattr(
        social_media,
        "create_minio_signed_clip_url",
        lambda storage_path, *, clip_id=None, expires_in_seconds=3600: calls.append(
            (storage_path, clip_id, expires_in_seconds)
        )
        or "https://storage.example/signed",
    )

    signed_url = social_media.create_signed_clip_url(
        "generated-clips/clip-1.mp4",
        expires_in_seconds=90,
    )

    assert signed_url == "https://storage.example/signed"
    assert calls == [("generated-clips/clip-1.mp4", "clip-1", 90)]


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

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yt_dlp

from config import (
    YTDLP_DOWNLOAD_RETRIES,
    YTDLP_EXTRACTOR_RETRIES,
    YTDLP_FRAGMENT_RETRIES,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
)
from services import video_downloader as video_downloader_module
from services.video_downloader import VideoDownloader
from tasks.clips.helpers import source_video
from utils import minio_client as minio_client_module


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


class _NeverAcquireRedis:
    def set(self, *args, **kwargs):
        del args, kwargs
        return False

    def get(self, *_args, **_kwargs):
        return None

    def delete(self, *_args, **_kwargs):
        return 0


@pytest.fixture(autouse=True)
def _stub_source_probe_update(monkeypatch):
    monkeypatch.setattr(source_video, "build_video_source_probe_update", lambda _path: {})


def _expected_profiled_cache_path(tmp_path: Path, video_id: str, suffix: str = ".mp4") -> str:
    return str(
        Path(tmp_path)
        / "cache"
        / f"{video_id}__{source_video._default_source_profile(source_video._UNSET_SOURCE_MAX_HEIGHT)}{suffix}"
    )


def test_resolve_source_video_deduplicates_download_for_concurrent_same_video(
    monkeypatch,
    tmp_path: Path,
):
    shared_row = {
        "raw_video_path": None,
        "url": "https://example.com/watch?v=abc123",
    }
    row_lock = threading.Lock()
    fake_redis = _FakeRedis()
    download_calls: list[str] = []

    class _FakeDownloader:
        def download(self, _url: str, video_id: str):
            download_calls.append(video_id)
            # Keep the lock owned briefly so another thread must wait.
            time.sleep(0.08)
            output_path = tmp_path / f"{video_id}.mp4"
            output_path.write_bytes(b"video")
            return {
                "path": str(output_path),
                "duration": 120,
            }

    def _load_video_row(_video_id: str) -> dict:
        with row_lock:
            return dict(shared_row)

    def _persist_raw_video_path(_video_id: str, raw_path: str):
        with row_lock:
            shared_row["raw_video_path"] = raw_path

    monkeypatch.setattr(source_video, "get_redis_connection", lambda: fake_redis)
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 5)
    monkeypatch.setattr(source_video, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-1.mp4", "etag-1"),
    )

    results: dict[str, source_video.SourceVideoResolution] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _run(worker_name: str):
        try:
            barrier.wait(timeout=2)
            resolution = source_video.resolve_source_video(
                video_id="video-1",
                source_url=shared_row["url"],
                initial_raw_video_path=None,
                initial_raw_video_storage_path=None,
                downloader=_FakeDownloader(),
                load_video_row=_load_video_row,
                persist_raw_video_path=_persist_raw_video_path,
                logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
                job_id=worker_name,
                on_wait_for_download=lambda _elapsed: None,
                on_download_start=lambda: None,
            )
            results[worker_name] = resolution
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=("job-a",), daemon=True)
    t2 = threading.Thread(target=_run, args=("job-b",), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert not errors
    assert len(download_calls) == 1
    assert len(results) == 2
    strategies = {resolution.strategy for resolution in results.values()}
    assert "downloaded_now" in strategies
    assert "waited_and_reused" in strategies
    assert any(resolution.wait_seconds > 0 for resolution in results.values())


def test_resolve_source_video_reuses_existing_raw_path_without_download(
    monkeypatch,
    tmp_path: Path,
):
    existing_path = tmp_path / "ready.mp4"
    existing_path.write_bytes(b"ok")
    download_called = {"value": False}

    class _FakeDownloader:
        def download(self, *_args, **_kwargs):
            download_called["value"] = True
            raise AssertionError("download should not be called")

    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1080, 1920))

    result = source_video.resolve_source_video(
        video_id="video-2",
        source_url="https://example.com/watch?v=xyz",
        initial_raw_video_path=str(existing_path),
        initial_raw_video_storage_path=None,
        downloader=_FakeDownloader(),
        load_video_row=lambda _video_id: {
            "raw_video_path": str(existing_path),
            "url": "https://example.com/watch?v=xyz",
        },
        persist_raw_video_path=lambda *_args, **_kwargs: None,
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        job_id="job-1",
    )

    assert not download_called["value"]
    assert result.strategy == "reused_existing"
    assert result.video_path == str(existing_path)


def test_resolve_source_video_prefers_fresh_download_when_requested(
    monkeypatch,
    tmp_path: Path,
):
    existing_path = tmp_path / "cached.mp4"
    existing_path.write_bytes(b"cached")
    fresh_path = tmp_path / "fresh.mp4"
    persisted_path = {"value": None}
    download_called = {"value": False}

    class _FakeDownloader:
        def download(self, _url: str, _video_id: str):
            download_called["value"] = True
            fresh_path.write_bytes(b"fresh")
            return {"path": str(fresh_path), "duration": 60}

    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "get_redis_connection", lambda: None)
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-fresh.mp4", "etag-fresh"),
    )

    result = source_video.resolve_source_video(
        video_id="video-fresh",
        source_url="https://example.com/watch?v=fresh",
        initial_raw_video_path=str(existing_path),
        initial_raw_video_storage_path=None,
        downloader=_FakeDownloader(),
        load_video_row=lambda _video_id: {
            "raw_video_path": str(existing_path),
            "url": "https://example.com/watch?v=fresh",
        },
        persist_raw_video_path=lambda _video_id, raw_path: persisted_path.__setitem__(
            "value", raw_path
        ),
        logger=type(
            "Logger",
            (),
            {
                "warning": lambda *args, **kwargs: None,
                "info": lambda *args, **kwargs: None,
            },
        )(),
        job_id="job-fresh",
        prefer_fresh_download=True,
    )

    assert download_called["value"] is True
    assert persisted_path["value"] == str(fresh_path)
    assert result.strategy == "downloaded_now"
    assert result.video_path == str(fresh_path)


def test_resolve_source_video_can_request_unbounded_source_height(
    monkeypatch,
    tmp_path: Path,
):
    captured_max_height = {"value": "unset"}
    fresh_path = tmp_path / "fresh-best.mp4"

    class _FakeDownloader:
        def download(self, _url: str, _video_id: str, *, max_height: int | None = 1080):
            captured_max_height["value"] = max_height
            fresh_path.write_bytes(b"fresh")
            return {"path": str(fresh_path), "duration": 60}

    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "get_redis_connection", lambda: None)
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-best-source.mp4", "etag-best"),
    )

    result = source_video.resolve_source_video(
        video_id="video-best-source",
        source_url="https://example.com/watch?v=best",
        initial_raw_video_path=None,
        initial_raw_video_storage_path=None,
        downloader=_FakeDownloader(),
        load_video_row=lambda _video_id: {"raw_video_path": None, "url": None},
        persist_raw_video_path=lambda *_args, **_kwargs: None,
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        job_id="job-best-source",
        source_max_height=None,
        prefer_fresh_download=True,
    )

    assert captured_max_height["value"] is None
    assert result.video_path == str(fresh_path)


def test_resolve_source_video_downloads_from_canonical_storage_before_url(
    monkeypatch,
    tmp_path: Path,
):
    class _FailIfDownloaded:
        def download(self, *_args, **_kwargs):
            raise AssertionError("URL download should not be used when canonical storage is available")

    class _FakeMinioClient:
        def fget_object(self, _bucket: str, _object_name: str, target_path: str):
            Path(target_path).write_bytes(b"stored-video")

    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1280, 720))
    monkeypatch.setattr(minio_client_module, "get_minio_client", lambda: _FakeMinioClient())

    result = source_video.resolve_source_video(
        video_id="video-storage-first",
        source_url="https://example.com/watch?v=storage",
        initial_raw_video_path=None,
        initial_raw_video_storage_path="raw/video-storage-first.mp4",
        downloader=_FailIfDownloaded(),
        load_video_row=lambda _video_id: {
            "raw_video_path": None,
            "raw_video_storage_path": "raw/video-storage-first.mp4",
            "url": "https://example.com/watch?v=storage",
        },
        persist_raw_video_path=lambda *_args, **_kwargs: None,
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
        job_id="job-storage-first",
    )

    assert result.strategy == "downloaded_from_storage"
    assert Path(result.video_path).is_file()


def test_resolve_source_video_prefers_fresh_but_reuses_waited_download(
    monkeypatch,
    tmp_path: Path,
):
    stale_path = tmp_path / "stale.mp4"
    stale_path.write_bytes(b"stale")
    fresh_path = tmp_path / "fresh.mp4"
    shared_row = {
        "raw_video_path": str(stale_path),
        "url": "https://example.com/watch?v=fresh-wait",
    }
    row_lock = threading.Lock()
    fake_redis = _FakeRedis()
    download_calls: list[str] = []

    class _FakeDownloader:
        def download(self, _url: str, video_id: str):
            download_calls.append(video_id)
            time.sleep(0.08)
            fresh_path.write_bytes(b"fresh")
            return {
                "path": str(fresh_path),
                "duration": 120,
            }

    def _load_video_row(_video_id: str) -> dict:
        with row_lock:
            return dict(shared_row)

    def _persist_raw_video_path(_video_id: str, raw_path: str):
        with row_lock:
            shared_row["raw_video_path"] = raw_path

    monkeypatch.setattr(source_video, "get_redis_connection", lambda: fake_redis)
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 5)
    monkeypatch.setattr(source_video, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-fresh-wait.mp4", "etag-fresh-wait"),
    )

    results: dict[str, source_video.SourceVideoResolution] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _run(worker_name: str):
        try:
            barrier.wait(timeout=2)
            resolution = source_video.resolve_source_video(
                video_id="video-fresh-wait",
                source_url=shared_row["url"],
                initial_raw_video_path=str(stale_path),
                initial_raw_video_storage_path=None,
                downloader=_FakeDownloader(),
                load_video_row=_load_video_row,
                persist_raw_video_path=_persist_raw_video_path,
                logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
                job_id=worker_name,
                prefer_fresh_download=True,
                on_wait_for_download=lambda _elapsed: None,
                on_download_start=lambda: None,
            )
            results[worker_name] = resolution
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    t1 = threading.Thread(target=_run, args=("job-a",), daemon=True)
    t2 = threading.Thread(target=_run, args=("job-b",), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert not errors
    assert len(download_calls) == 1
    assert len(results) == 2
    strategies = {resolution.strategy for resolution in results.values()}
    assert "downloaded_now" in strategies
    assert "waited_and_reused" in strategies
    reused = [resolution for resolution in results.values() if resolution.strategy == "waited_and_reused"]
    assert reused
    assert reused[0].video_path == _expected_profiled_cache_path(tmp_path, "video-fresh-wait")
    assert Path(reused[0].video_path).is_file()


def test_resolve_source_video_prefers_fresh_with_three_waiters_uses_first_fresh_result(
    monkeypatch,
    tmp_path: Path,
):
    shared_row = {
        "raw_video_path": None,
        "raw_video_storage_path": None,
        "url": "https://example.com/watch?v=triple-fresh",
    }
    row_lock = threading.Lock()
    fake_redis = _FakeRedis()
    download_calls: list[str] = []

    class _FakeDownloader:
        def download(self, _url: str, video_id: str):
            download_calls.append(video_id)
            time.sleep(0.08)
            output_path = tmp_path / f"{threading.current_thread().name}-{video_id}.mp4"
            output_path.write_bytes(b"video")
            return {
                "path": str(output_path),
                "duration": 120,
            }

    def _load_video_row(_video_id: str) -> dict:
        with row_lock:
            return dict(shared_row)

    def _persist_raw_video_path(_video_id: str, raw_path: str):
        with row_lock:
            shared_row["raw_video_path"] = raw_path
        time.sleep(0.2)

    def _persist_raw_video_metadata(_video_id: str, payload: dict):
        with row_lock:
            shared_row.update(payload)
        time.sleep(0.2)

    monkeypatch.setattr(source_video, "get_redis_connection", lambda: fake_redis)
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 5)
    monkeypatch.setattr(source_video, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-triple-fresh.mp4", "etag-triple-fresh"),
    )

    results: dict[str, source_video.SourceVideoResolution] = {}
    errors: list[Exception] = []

    def _run(worker_name: str, delay: float):
        try:
            if delay:
                time.sleep(delay)
            resolution = source_video.resolve_source_video(
                video_id="video-triple-fresh",
                source_url=shared_row["url"],
                initial_raw_video_path=None,
                initial_raw_video_storage_path=None,
                downloader=_FakeDownloader(),
                load_video_row=_load_video_row,
                persist_raw_video_path=_persist_raw_video_path,
                persist_raw_video_metadata=_persist_raw_video_metadata,
                logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
                job_id=worker_name,
                prefer_fresh_download=True,
                on_wait_for_download=lambda _elapsed: None,
                on_download_start=lambda: None,
            )
            results[worker_name] = resolution
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    threads = [
        threading.Thread(target=_run, name="job-a", args=("job-a", 0.0), daemon=True),
        threading.Thread(target=_run, name="job-b", args=("job-b", 0.02), daemon=True),
        threading.Thread(target=_run, name="job-c", args=("job-c", 0.12), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert len(download_calls) == 1
    assert len(results) == 3
    waited = [
        resolution for resolution in results.values() if resolution.strategy == "waited_and_reused"
    ]
    assert len(waited) == 2
    expected_cache_path = _expected_profiled_cache_path(tmp_path, "video-triple-fresh")
    assert {resolution.video_path for resolution in waited} == {expected_cache_path}
    assert Path(expected_cache_path).is_file()


def test_resolve_source_video_waiter_reuses_shared_cache_not_producer_workdir(
    monkeypatch,
    tmp_path: Path,
):
    shared_row = {
        "raw_video_path": None,
        "raw_video_storage_path": None,
        "url": "https://example.com/watch?v=durable-handoff",
    }
    row_lock = threading.Lock()
    fake_redis = _FakeRedis()
    raw_path_persisted = threading.Event()
    producer_paths: list[str] = []

    class _FakeDownloader:
        def download(self, _url: str, video_id: str):
            output_path = tmp_path / f"{threading.current_thread().name}-{video_id}.mp4"
            output_path.write_bytes(b"producer-video")
            producer_paths.append(str(output_path))
            return {
                "path": str(output_path),
                "duration": 120,
            }

    def _load_video_row(_video_id: str) -> dict:
        with row_lock:
            return dict(shared_row)

    def _persist_raw_video_path(_video_id: str, raw_path: str):
        with row_lock:
            shared_row["raw_video_path"] = raw_path
        raw_path_persisted.set()
        time.sleep(0.2)

    def _persist_raw_video_metadata(_video_id: str, payload: dict):
        with row_lock:
            shared_row.update(payload)

    monkeypatch.setattr(source_video, "get_redis_connection", lambda: fake_redis)
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr(source_video, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 5)
    monkeypatch.setattr(source_video, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        source_video,
        "upload_raw_video_to_storage",
        lambda **_kwargs: ("raw/video-durable-handoff.mp4", "etag-durable-handoff"),
    )

    results: dict[str, source_video.SourceVideoResolution] = {}
    errors: list[Exception] = []

    def _run_producer():
        try:
            results["producer"] = source_video.resolve_source_video(
                video_id="video-durable-handoff",
                source_url=shared_row["url"],
                initial_raw_video_path=None,
                initial_raw_video_storage_path=None,
                downloader=_FakeDownloader(),
                load_video_row=_load_video_row,
                persist_raw_video_path=_persist_raw_video_path,
                persist_raw_video_metadata=_persist_raw_video_metadata,
                logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
                job_id="job-producer",
                prefer_fresh_download=True,
                on_wait_for_download=lambda _elapsed: None,
                on_download_start=lambda: None,
            )
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    def _run_waiter():
        try:
            assert raw_path_persisted.wait(timeout=1)
            results["waiter"] = source_video.resolve_source_video(
                video_id="video-durable-handoff",
                source_url=shared_row["url"],
                initial_raw_video_path=None,
                initial_raw_video_storage_path=None,
                downloader=_FakeDownloader(),
                load_video_row=_load_video_row,
                persist_raw_video_path=_persist_raw_video_path,
                persist_raw_video_metadata=_persist_raw_video_metadata,
                logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
                job_id="job-waiter",
                prefer_fresh_download=True,
                on_wait_for_download=lambda _elapsed: None,
                on_download_start=lambda: None,
            )
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    producer = threading.Thread(target=_run_producer, daemon=True)
    waiter = threading.Thread(target=_run_waiter, daemon=True)
    producer.start()
    waiter.start()
    producer.join(timeout=3)
    waiter.join(timeout=3)

    assert not errors
    assert results["producer"].strategy == "downloaded_now"
    assert results["waiter"].strategy == "waited_and_reused"
    expected_cache_path = _expected_profiled_cache_path(tmp_path, "video-durable-handoff")
    assert results["waiter"].video_path == expected_cache_path
    assert results["waiter"].video_path != producer_paths[0]
    assert Path(expected_cache_path).read_bytes() == b"producer-video"


def test_resolve_source_video_times_out_when_lock_is_never_released(monkeypatch):
    wait_events = {"count": 0}

    class _FakeDownloader:
        def download(self, *_args, **_kwargs):
            raise AssertionError("download should not be reached on lock timeout")

    monkeypatch.setattr(source_video, "get_redis_connection", lambda: _NeverAcquireRedis())
    monkeypatch.setattr(source_video, "SOURCE_VIDEO_LOCK_WAIT_SECONDS", 1)
    monkeypatch.setattr(source_video, "_WAIT_POLL_INTERVAL_SECONDS", 0.01)

    with pytest.raises(RuntimeError, match="Timed out waiting for source video download lock"):
        source_video.resolve_source_video(
            video_id="video-timeout",
            source_url="https://example.com/video",
            initial_raw_video_path=None,
            initial_raw_video_storage_path=None,
            downloader=_FakeDownloader(),
            load_video_row=lambda _video_id: {"raw_video_path": None, "url": None},
            persist_raw_video_path=lambda *_args, **_kwargs: None,
            logger=type("Logger", (), {"warning": lambda *args, **kwargs: None})(),
            job_id="job-timeout",
            on_wait_for_download=lambda _elapsed: wait_events.__setitem__(
                "count", wait_events["count"] + 1
            ),
        )

    assert wait_events["count"] > 0


def test_video_downloader_base_options_include_hardening_flags():
    opts = VideoDownloader()._base_ydl_opts()
    assert opts["format"] == VideoDownloader._format_selector_for_max_height(1080)
    assert "height<=720" not in opts["format"]
    assert opts["noplaylist"] is True
    assert "extractor_args" not in opts
    assert opts["socket_timeout"] == YTDLP_SOCKET_TIMEOUT_SECONDS
    assert opts["retries"] == YTDLP_DOWNLOAD_RETRIES
    assert opts["fragment_retries"] == YTDLP_FRAGMENT_RETRIES
    assert opts["extractor_retries"] == YTDLP_EXTRACTOR_RETRIES


def test_video_downloader_uses_writable_runtime_cookie_file(monkeypatch, tmp_path: Path):
    runtime_cookie = tmp_path / "runtime-cookies.txt"
    runtime_cookie.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_SOURCE_FILE", None)
    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_FILE", str(runtime_cookie))

    downloader = VideoDownloader(work_dir=str(tmp_path / "work"))
    opts = downloader._base_ydl_opts()

    assert opts["cookiefile"] == str(runtime_cookie)


def test_video_downloader_copies_read_only_cookie_source_to_runtime(monkeypatch, tmp_path: Path):
    legacy_cookie = tmp_path / "legacy-cookies.txt"
    legacy_cookie.write_text("# Netscape HTTP Cookie File\nlegacy\n", encoding="utf-8")

    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_SOURCE_FILE", None)
    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_FILE", str(legacy_cookie))
    monkeypatch.setattr(
        VideoDownloader,
        "_is_writable_cookie_path",
        staticmethod(lambda path: str(path) != str(legacy_cookie)),
    )

    downloader = VideoDownloader(work_dir=str(tmp_path / "work"))
    opts = downloader._base_ydl_opts()
    copied_cookie = Path(opts["cookiefile"])

    assert copied_cookie != legacy_cookie
    assert copied_cookie.read_text(encoding="utf-8") == legacy_cookie.read_text(
        encoding="utf-8"
    )


def test_video_downloader_omits_cookiefile_when_no_cookies_configured(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_SOURCE_FILE", None)
    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_FILE", None)

    downloader = VideoDownloader(work_dir=str(tmp_path))
    opts = downloader._base_ydl_opts()

    assert "cookiefile" not in opts


def test_video_downloader_probe_does_not_send_cookies(monkeypatch, tmp_path: Path):
    source_cookie = tmp_path / "source-cookies.txt"
    source_cookie.write_text("# Netscape HTTP Cookie File\nsource\n", encoding="utf-8")
    runtime_cookie = tmp_path / "runtime" / "cookies.txt"
    seen: dict[str, object] = {}

    class _FakeYDL:
        def __init__(self, opts: dict):
            seen["cookiefile"] = opts.get("cookiefile")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _url: str, *, download: bool):
            assert download is False
            return {
                "duration": 120,
                "formats": [{"acodec": "mp4a.40.2"}],
                "extractor_key": "youtube",
                "id": "abc123",
            }

    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_SOURCE_FILE", str(source_cookie))
    monkeypatch.setattr(video_downloader_module, "YTDLP_COOKIES_FILE", str(runtime_cookie))
    monkeypatch.setattr(video_downloader_module.yt_dlp, "YoutubeDL", _FakeYDL)
    monkeypatch.setattr(video_downloader_module, "_validate_url", lambda _url: None)

    downloader = VideoDownloader(work_dir=str(tmp_path / "work"))
    probe = downloader.probe_url("https://www.youtube.com/watch?v=abc123")

    assert probe["can_download"] is True
    assert seen["cookiefile"] is None


def test_video_downloader_resolve_output_path_prefers_video_sidecar_over_newer_audio(
    monkeypatch,
    tmp_path: Path,
):
    expected = tmp_path / "video-123.mp4"
    split_video = tmp_path / "video-123.f137.mp4"
    split_audio = tmp_path / "video-123.f140.m4a"
    split_video.write_bytes(b"video")
    split_audio.write_bytes(b"audio")
    mtimes = {str(split_video): 1.0, str(split_audio): 2.0}

    monkeypatch.setattr(
        video_downloader_module.os.path,
        "getmtime",
        lambda path: mtimes[str(path)],
    )
    monkeypatch.setattr(
        VideoDownloader,
        "_has_video_stream",
        staticmethod(lambda path: str(path) == str(split_video)),
    )

    resolved = VideoDownloader._resolve_output_path(str(expected))
    assert resolved == str(split_video)


def test_video_downloader_resolve_output_path_skips_audio_only_expected_file(
    monkeypatch,
    tmp_path: Path,
):
    expected = tmp_path / "video-123.mp4"
    split_video = tmp_path / "video-123.f137.mp4"
    expected.write_bytes(b"audio-only")
    split_video.write_bytes(b"video")
    mtimes = {str(expected): 2.0, str(split_video): 1.0}

    monkeypatch.setattr(
        video_downloader_module.os.path,
        "getmtime",
        lambda path: mtimes[str(path)],
    )
    monkeypatch.setattr(
        VideoDownloader,
        "_has_video_stream",
        staticmethod(lambda path: str(path) == str(split_video)),
    )

    resolved = VideoDownloader._resolve_output_path(str(expected))
    assert resolved == str(split_video)


def test_video_downloader_fallback_selector_is_not_720_limited(
    monkeypatch,
    tmp_path: Path,
):
    selectors: list[tuple[str, str]] = []

    def _fake_download_with_opts(
        self,
        url: str,
        output_path: str,
        *,
        format_selector: str,
        attempt_label: str,
        max_height: int | None = 1080,
        ydl_overrides: dict[str, object] | None = None,
    ):
        del self, url, output_path, max_height, ydl_overrides
        selectors.append((attempt_label, format_selector))
        return {
            "title": "Video",
            "duration": 10,
            "thumbnail": None,
            "extractor_key": "youtube",
            "id": "abc123",
        }

    audio_checks = iter([False, True])
    monkeypatch.setattr(video_downloader_module, "_validate_url", lambda _url: None)
    monkeypatch.setattr(VideoDownloader, "_download_with_opts", _fake_download_with_opts)
    monkeypatch.setattr(VideoDownloader, "_resolve_output_path", lambda _self, path: path)
    monkeypatch.setattr(VideoDownloader, "_has_audio_stream", lambda _self, _path: next(audio_checks))

    downloader = VideoDownloader(work_dir=str(tmp_path))
    result = downloader.download("https://example.com/video", "video-123")

    assert result["path"].endswith("video-123.mp4")
    assert selectors[0] == (
        "best_source_default",
        VideoDownloader._format_selector_for_max_height(1080),
    )
    assert selectors[1][0] == "compatibility_fallback"
    assert "height<=720" not in selectors[1][1]


def test_video_downloader_retries_legacy_clients_for_youtube_on_primary_error(
    monkeypatch,
    tmp_path: Path,
):
    attempts: list[tuple[str, dict[str, object] | None]] = []

    def _fake_download_with_opts(
        self,
        url: str,
        output_path: str,
        *,
        format_selector: str,
        attempt_label: str,
        max_height: int | None = 1080,
        ydl_overrides: dict[str, object] | None = None,
    ):
        del self, url, output_path, format_selector, max_height
        attempts.append((attempt_label, ydl_overrides))
        if attempt_label == "best_source_default":
            raise yt_dlp.utils.DownloadError("primary failed")
        return {
            "title": "Video",
            "duration": 10,
            "thumbnail": None,
            "extractor_key": "youtube",
            "id": "abc123",
        }

    monkeypatch.setattr(video_downloader_module, "_validate_url", lambda _url: None)
    monkeypatch.setattr(VideoDownloader, "_download_with_opts", _fake_download_with_opts)
    monkeypatch.setattr(VideoDownloader, "_resolve_output_path", lambda _self, path: path)
    monkeypatch.setattr(VideoDownloader, "_has_audio_stream", lambda _self, _path: True)

    downloader = VideoDownloader(work_dir=str(tmp_path))
    result = downloader.download("https://www.youtube.com/watch?v=abc123", "video-123")

    assert result["path"].endswith("video-123.mp4")
    assert attempts[0] == ("best_source_default", None)
    assert attempts[1] == (
        "best_source_legacy_clients",
        {"extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}}},
    )
    assert len(attempts) == 2


def test_video_downloader_no_legacy_retry_when_primary_succeeds(
    monkeypatch,
    tmp_path: Path,
):
    attempts: list[tuple[str, dict[str, object] | None]] = []

    def _fake_download_with_opts(
        self,
        url: str,
        output_path: str,
        *,
        format_selector: str,
        attempt_label: str,
        max_height: int | None = 1080,
        ydl_overrides: dict[str, object] | None = None,
    ):
        del self, url, output_path, format_selector, max_height
        attempts.append((attempt_label, ydl_overrides))
        return {
            "title": "Video",
            "duration": 10,
            "thumbnail": None,
            "extractor_key": "youtube",
            "id": "abc123",
        }

    monkeypatch.setattr(video_downloader_module, "_validate_url", lambda _url: None)
    monkeypatch.setattr(VideoDownloader, "_download_with_opts", _fake_download_with_opts)
    monkeypatch.setattr(VideoDownloader, "_resolve_output_path", lambda _self, path: path)
    monkeypatch.setattr(VideoDownloader, "_has_audio_stream", lambda _self, _path: True)

    downloader = VideoDownloader(work_dir=str(tmp_path))
    result = downloader.download("https://www.youtube.com/watch?v=abc123", "video-123")

    assert result["path"].endswith("video-123.mp4")
    assert attempts == [("best_source_default", None)]

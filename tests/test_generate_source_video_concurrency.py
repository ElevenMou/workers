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

    class _FakeStorageBucket:
        def download(self, _path: str):
            return b"stored-video"

    class _FakeStorageClient:
        def from_(self, _bucket: str):
            return _FakeStorageBucket()

    monkeypatch.setattr(source_video, "RAW_VIDEO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(source_video, "probe_video_size", lambda _path: (1280, 720))
    monkeypatch.setattr(source_video, "supabase", type("S", (), {"storage": _FakeStorageClient()})())

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
    assert reused[0].video_path == str(fresh_path)


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
        "primary_av_merge_default",
        VideoDownloader._format_selector_for_max_height(1080),
    )
    assert selectors[1][0] == "fallback_muxed_av"
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
        if attempt_label == "primary_av_merge_default":
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
    assert attempts[0] == ("primary_av_merge_default", None)
    assert attempts[1] == (
        "primary_av_merge_legacy_clients",
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
    assert attempts == [("primary_av_merge_default", None)]

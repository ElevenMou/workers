"""Shared source-video resolution with same-video download deduplication."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from config import (
    MINIO_RAW_VIDEOS_BUCKET,
    RAW_VIDEO_CACHE_DIR,
    SOURCE_VIDEO_LOCK_TTL_SECONDS,
    SOURCE_VIDEO_LOCK_WAIT_SECONDS,
)
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.media import probe_video_size
from utils.redis_client import get_redis_connection
from utils.media_storage import (
    preferred_source_video_order,
    upload_raw_video,
)
from utils.supabase_client import supabase

_VIDEO_RAW_TTL_HOURS = 24
_WAIT_STAGE_NOTIFY_INTERVAL_SECONDS = 5.0
_WAIT_POLL_INTERVAL_SECONDS = 1.0
_UNSET_SOURCE_MAX_HEIGHT = object()
_RAW_VIDEO_STORAGE_BUCKET = MINIO_RAW_VIDEOS_BUCKET


@dataclass(slots=True)
class SourceVideoResolution:
    video_path: str
    width: int
    height: int
    strategy: str
    wait_seconds: float
    download_seconds: float
    download_metadata: dict[str, Any] | None = None
    storage_path: str | None = None
    storage_etag: str | None = None


def _decode_redis_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _release_lock_if_owned(*, conn, key: str, token: str) -> None:
    try:
        current = conn.get(key)
    except Exception:
        return
    if current is None:
        return
    if _decode_redis_value(current) != token:
        return
    try:
        conn.delete(key)
    except Exception:
        return


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, (str, bytes)):
        return None
    text = str(value).strip() if value else ""
    return text or None


def _paths_match(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.abspath(left) == os.path.abspath(right)
    except Exception:
        return left == right


def _path_is_fresh(candidate: str | None, entry_snapshot: str | None) -> bool:
    if not candidate:
        return False
    if not entry_snapshot:
        return True
    return not _paths_match(candidate, entry_snapshot)


def _value_is_fresh(candidate: str | None, entry_snapshot: str | None) -> bool:
    if not candidate:
        return False
    if not entry_snapshot:
        return True
    return candidate != entry_snapshot


def _should_force_storage_cache_refresh(
    *,
    latest_storage_path: str | None,
    entry_raw_storage_path: str | None,
    prefer_fresh_download: bool,
) -> bool:
    if not prefer_fresh_download or not latest_storage_path:
        return False
    if entry_raw_storage_path:
        return latest_storage_path != entry_raw_storage_path
    # When a waited job had no storage snapshot at entry time, a cache file
    # keyed only by video_id may still contain an older raw source. Refresh it
    # from canonical storage before reusing the artifact.
    return True


def _raw_cache_path(video_id: str) -> str:
    base = Path(RAW_VIDEO_CACHE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{video_id}.mp4")


def upload_raw_video_to_storage(
    *,
    video_id: str,
    local_video_path: str,
    logger,
    job_id: str,
) -> tuple[str | None, str | None]:
    """Upload canonical raw source into private storage and return path + hash."""
    return upload_raw_video(
        video_id=video_id,
        local_video_path=local_video_path,
        logger=logger,
        job_id=job_id,
    )


def _resolve_storage_video_path(
    *,
    video_id: str,
    job_id: str,
    logger,
    raw_video_storage_path: str | None,
    force_refresh_cache: bool = False,
) -> tuple[str, int, int] | None:
    if not raw_video_storage_path:
        return None

    cache_path = _raw_cache_path(video_id)
    if not force_refresh_cache:
        cached_existing = _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=cache_path,
        )
        if cached_existing is not None:
            return cached_existing

    try:
        from utils.minio_client import get_minio_client

        client = get_minio_client()
        object_name = str(raw_video_storage_path).strip().replace("\\", "/").lstrip("/")
        bucket_prefix = f"{_RAW_VIDEO_STORAGE_BUCKET}/"
        if object_name.startswith(bucket_prefix):
            object_name = object_name[len(bucket_prefix) :]
        tmp_path = f"{cache_path}.tmp"
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(_RAW_VIDEO_STORAGE_BUCKET, object_name, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        logger.warning(
            "[%s] Could not download canonical raw source %s/%s: %s",
            job_id,
            _RAW_VIDEO_STORAGE_BUCKET,
            raw_video_storage_path,
            exc,
        )
        return None

    return _resolve_existing_video_path(
        video_id=video_id,
        job_id=job_id,
        logger=logger,
        raw_video_path=cache_path,
    )


def _resolve_existing_video_path(
    *,
    video_id: str,
    job_id: str,
    logger,
    raw_video_path: str | None,
) -> tuple[str, int, int] | None:
    if not raw_video_path:
        return None
    if not os.path.isfile(raw_video_path):
        return None
    try:
        width, height = probe_video_size(raw_video_path)
    except Exception as exc:
        logger.warning(
            "[%s] Existing raw video is unusable for %s (%s): %s",
            job_id,
            video_id,
            raw_video_path,
            exc,
        )
        return None
    return raw_video_path, width, height


def _materialize_waited_local_video_to_cache(
    *,
    video_id: str,
    job_id: str,
    logger,
    raw_video_path: str | None,
) -> tuple[str, int, int] | None:
    normalized_path = _normalize_optional_text(raw_video_path)
    if normalized_path is None:
        return None

    cache_path = _raw_cache_path(video_id)
    if _paths_match(normalized_path, cache_path):
        return _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=cache_path,
        )

    source_existing = _resolve_existing_video_path(
        video_id=video_id,
        job_id=job_id,
        logger=logger,
        raw_video_path=normalized_path,
    )
    if source_existing is None:
        return None

    tmp_cache_path = f"{cache_path}.tmp.{uuid.uuid4().hex}"
    try:
        shutil.copyfile(normalized_path, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
    except Exception as exc:
        logger.warning(
            "[%s] Failed to materialize waited raw video to shared cache for %s (%s -> %s): %s",
            job_id,
            video_id,
            normalized_path,
            cache_path,
            exc,
        )
        try:
            if os.path.exists(tmp_cache_path):
                os.remove(tmp_cache_path)
        except OSError:
            pass
        return None

    cached_existing = _resolve_existing_video_path(
        video_id=video_id,
        job_id=job_id,
        logger=logger,
        raw_video_path=cache_path,
    )
    if cached_existing is not None:
        return cached_existing

    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
    except OSError:
        pass
    return None


def _try_resolve_waited_video(
    *,
    video_id: str,
    job_id: str,
    logger,
    latest_path: str | None,
    latest_storage_path: str | None,
    entry_raw_video_path: str | None,
    entry_raw_storage_path: str | None,
    prefer_fresh_download: bool,
    waited_seconds: float,
) -> SourceVideoResolution | None:
    """Try to resolve a source video from storage or local cache after waiting.

    Shared by both the in-loop and post-loop resolution paths.
    """
    def _resolve_storage_candidate() -> SourceVideoResolution | None:
        if prefer_fresh_download and not _value_is_fresh(
            latest_storage_path,
            entry_raw_storage_path,
        ):
            return None

        storage_is_fresh = _should_force_storage_cache_refresh(
            latest_storage_path=latest_storage_path,
            entry_raw_storage_path=entry_raw_storage_path,
            prefer_fresh_download=prefer_fresh_download,
        )
        latest_storage_existing = _resolve_storage_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_storage_path=latest_storage_path,
            force_refresh_cache=storage_is_fresh,
        )
        if latest_storage_existing is None:
            return None

        path, width, height = latest_storage_existing
        return SourceVideoResolution(
            video_path=path,
            width=width,
            height=height,
            strategy="waited_and_downloaded_from_storage",
            wait_seconds=float(max(0.0, waited_seconds)),
            download_seconds=0.0,
            download_metadata=None,
            storage_path=latest_storage_path,
        )

    def _resolve_local_candidate() -> SourceVideoResolution | None:
        if prefer_fresh_download and not _path_is_fresh(latest_path, entry_raw_video_path):
            return None

        latest_existing = _materialize_waited_local_video_to_cache(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=latest_path,
        )
        if latest_existing is None:
            return None

        path, width, height = latest_existing
        return SourceVideoResolution(
            video_path=path,
            width=width,
            height=height,
            strategy="waited_and_reused",
            wait_seconds=float(max(0.0, waited_seconds)),
            download_seconds=0.0,
            download_metadata=None,
        )

    for candidate in preferred_source_video_order():
        resolved = (
            _resolve_local_candidate()
            if candidate == "local"
            else _resolve_storage_candidate()
        )
        if resolved is not None:
            return resolved

    return None


def _expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=_VIDEO_RAW_TTL_HOURS)).isoformat()


def resolve_source_video(
    *,
    video_id: str,
    source_url: str | None,
    initial_raw_video_path: str | None,
    initial_raw_video_storage_path: str | None,
    downloader: VideoDownloader,
    load_video_row: Callable[[str], dict[str, Any]],
    persist_raw_video_path: Callable[[str, str], None],
    persist_raw_video_metadata: Callable[[str, dict[str, Any]], None] | None = None,
    logger,
    job_id: str,
    source_max_height: int | None | object = _UNSET_SOURCE_MAX_HEIGHT,
    prefer_fresh_download: bool = False,
    on_wait_for_download: Callable[[float], None] | None = None,
    on_download_start: Callable[[], None] | None = None,
) -> SourceVideoResolution:
    """Resolve a usable source-video path with same-video download deduplication."""
    entry_raw_video_path = _normalize_optional_text(initial_raw_video_path)
    entry_raw_storage_path = _normalize_optional_text(initial_raw_video_storage_path)
    if not prefer_fresh_download:
        def _initial_local_candidate() -> SourceVideoResolution | None:
            existing = _resolve_existing_video_path(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                raw_video_path=initial_raw_video_path,
            )
            if existing is None:
                return None
            path, width, height = existing
            return SourceVideoResolution(
                video_path=path,
                width=width,
                height=height,
                strategy="reused_existing",
                wait_seconds=0.0,
                download_seconds=0.0,
                download_metadata=None,
            )

        def _initial_storage_candidate() -> SourceVideoResolution | None:
            storage_existing = _resolve_storage_video_path(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                raw_video_storage_path=entry_raw_storage_path,
            )
            if storage_existing is None:
                return None
            path, width, height = storage_existing
            return SourceVideoResolution(
                video_path=path,
                width=width,
                height=height,
                strategy="downloaded_from_storage",
                wait_seconds=0.0,
                download_seconds=0.0,
                download_metadata=None,
                storage_path=entry_raw_storage_path,
            )

        for candidate in preferred_source_video_order():
            resolved = (
                _initial_local_candidate()
                if candidate == "local"
                else _initial_storage_candidate()
            )
            if resolved is not None:
                return resolved

    wait_started_at = time.monotonic()
    lock_wait_seconds = max(1, int(SOURCE_VIDEO_LOCK_WAIT_SECONDS))
    lock_ttl_seconds = max(5, int(SOURCE_VIDEO_LOCK_TTL_SECONDS))
    lock_key = f"clipry:source-video:{video_id}"
    lock_token = f"{uuid.uuid4()}:{os.getpid()}"
    lock_acquired = False
    waited_for_lock = False
    last_wait_stage_emit = -_WAIT_STAGE_NOTIFY_INTERVAL_SECONDS

    try:
        conn = get_redis_connection()
    except Exception as exc:
        logger.warning(
            "[%s] Redis unavailable while resolving source for %s; falling back to direct download: %s",
            job_id,
            video_id,
            exc,
        )
        conn = None

    def _emit_wait_stage(elapsed_seconds: float) -> None:
        nonlocal last_wait_stage_emit
        if on_wait_for_download is None:
            return
        if (
            elapsed_seconds == 0.0
            or elapsed_seconds - last_wait_stage_emit >= _WAIT_STAGE_NOTIFY_INTERVAL_SECONDS
        ):
            on_wait_for_download(elapsed_seconds)
            last_wait_stage_emit = elapsed_seconds

    def _load_latest_path_url_and_storage() -> tuple[str | None, str | None, str | None]:
        row = load_video_row(video_id) or {}
        latest_path = _normalize_optional_text(row.get("raw_video_path"))
        latest_url = _normalize_optional_text(row.get("url"))
        latest_storage_path = _normalize_optional_text(row.get("raw_video_storage_path"))
        return (
            latest_path,
            latest_url,
            latest_storage_path,
        )

    try:
        while conn is not None and not lock_acquired:
            try:
                lock_acquired = bool(conn.set(lock_key, lock_token, nx=True, ex=lock_ttl_seconds))
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to acquire source-video lock for %s; falling back to direct download: %s",
                    job_id,
                    video_id,
                    exc,
                )
                conn = None
                break

            if lock_acquired:
                break

            waited_for_lock = True
            elapsed = time.monotonic() - wait_started_at
            _emit_wait_stage(float(elapsed))

            latest_path, _, latest_storage_path = _load_latest_path_url_and_storage()
            waited_resolution = _try_resolve_waited_video(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                latest_path=latest_path,
                latest_storage_path=latest_storage_path,
                entry_raw_video_path=entry_raw_video_path,
                entry_raw_storage_path=entry_raw_storage_path,
                prefer_fresh_download=prefer_fresh_download,
                waited_seconds=float(elapsed),
            )
            if waited_resolution is not None:
                return waited_resolution

            if elapsed >= lock_wait_seconds:
                raise RuntimeError(
                    "Timed out waiting for source video download lock "
                    f"(video_id={video_id}, waited={lock_wait_seconds}s)."
                )

            time.sleep(_WAIT_POLL_INTERVAL_SECONDS)

        latest_path, latest_url, latest_storage_path = _load_latest_path_url_and_storage()
        waited_seconds = time.monotonic() - wait_started_at
        if waited_for_lock:
            waited_resolution = _try_resolve_waited_video(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                latest_path=latest_path,
                latest_storage_path=latest_storage_path,
                entry_raw_video_path=entry_raw_video_path,
                entry_raw_storage_path=entry_raw_storage_path,
                prefer_fresh_download=prefer_fresh_download,
                waited_seconds=float(waited_seconds),
            )
            if waited_resolution is not None:
                return waited_resolution
        else:
            def _latest_local_candidate() -> SourceVideoResolution | None:
                if prefer_fresh_download and not _path_is_fresh(
                    latest_path,
                    entry_raw_video_path,
                ):
                    return None
                latest_existing = _resolve_existing_video_path(
                    video_id=video_id,
                    job_id=job_id,
                    logger=logger,
                    raw_video_path=latest_path,
                )
                if latest_existing is None:
                    return None
                path, width, height = latest_existing
                return SourceVideoResolution(
                    video_path=path,
                    width=width,
                    height=height,
                    strategy="reused_existing",
                    wait_seconds=float(max(0.0, waited_seconds)),
                    download_seconds=0.0,
                    download_metadata=None,
                )

            def _latest_storage_candidate() -> SourceVideoResolution | None:
                if prefer_fresh_download and not _value_is_fresh(
                    latest_storage_path,
                    entry_raw_storage_path,
                ):
                    return None
                latest_storage_existing = _resolve_storage_video_path(
                    video_id=video_id,
                    job_id=job_id,
                    logger=logger,
                    raw_video_storage_path=latest_storage_path,
                )
                if latest_storage_existing is None:
                    return None
                path, width, height = latest_storage_existing
                return SourceVideoResolution(
                    video_path=path,
                    width=width,
                    height=height,
                    strategy="downloaded_from_storage",
                    wait_seconds=float(max(0.0, waited_seconds)),
                    download_seconds=0.0,
                    download_metadata=None,
                    storage_path=latest_storage_path,
                )

            for candidate in preferred_source_video_order():
                resolved = (
                    _latest_local_candidate()
                    if candidate == "local"
                    else _latest_storage_candidate()
                )
                if resolved is not None:
                    return resolved

        resolved_url = str(source_url or latest_url or "").strip()
        if not resolved_url:
            raise RuntimeError(
                "Missing source URL and raw_video source metadata for clip generation"
            )

        if on_download_start is not None:
            on_download_start()

        download_started_at = time.monotonic()
        if source_max_height is _UNSET_SOURCE_MAX_HEIGHT:
            downloaded = downloader.download(resolved_url, video_id)
        else:
            try:
                downloaded = downloader.download(
                    resolved_url,
                    video_id,
                    max_height=source_max_height,
                )
            except TypeError:
                # Backward-compatible fallback for test doubles with legacy signature.
                downloaded = downloader.download(resolved_url, video_id)
        download_seconds = time.monotonic() - download_started_at

        downloaded_path = str(downloaded.get("path") or "").strip()
        downloaded_existing = _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=downloaded_path,
        )
        if downloaded_existing is None:
            raise RuntimeError(
                f"Downloaded source video is not usable for {video_id}: {downloaded_path or '<empty>'}"
            )

        raw_storage_path: str | None = None
        raw_storage_etag: str | None = None
        try:
            raw_storage_path, raw_storage_etag = upload_raw_video_to_storage(
                video_id=video_id,
                local_video_path=downloaded_path,
                logger=logger,
                job_id=job_id,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Canonical raw-source upload failed for %s: %s",
                job_id,
                video_id,
                exc,
            )

        metadata_update = build_raw_video_metadata_update(
            downloaded_path,
            raw_video_storage_path=raw_storage_path,
            raw_video_storage_etag=raw_storage_etag,
        )
        persist_raw_video_path(video_id, downloaded_path)
        if persist_raw_video_metadata is not None:
            persist_raw_video_metadata(video_id, metadata_update)
        waited_seconds = time.monotonic() - wait_started_at
        strategy = "downloaded_now"
        path, width, height = downloaded_existing
        return SourceVideoResolution(
            video_path=path,
            width=width,
            height=height,
            strategy=strategy,
            wait_seconds=float(max(0.0, waited_seconds)),
            download_seconds=float(max(0.0, download_seconds)),
            download_metadata=downloaded,
            storage_path=raw_storage_path,
            storage_etag=raw_storage_etag,
        )
    finally:
        if conn is not None and lock_acquired:
            _release_lock_if_owned(conn=conn, key=lock_key, token=lock_token)


def build_raw_video_metadata_update(
    raw_video_path: str | None,
    *,
    raw_video_storage_path: str | None = None,
    raw_video_storage_etag: str | None = None,
) -> dict[str, str | None]:
    payload: dict[str, str | None] = {
        "raw_video_path": raw_video_path,
        "raw_video_expires_at": _expires_at_iso(),
    }
    if raw_video_storage_path:
        payload["raw_video_storage_path"] = raw_video_storage_path
    if raw_video_storage_etag:
        payload["raw_video_storage_etag"] = raw_video_storage_etag
    return payload

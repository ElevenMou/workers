"""Shared source-video resolution with same-video download deduplication."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from config import (
    RAW_VIDEO_CACHE_DIR,
    SOURCE_VIDEO_LOCK_TTL_SECONDS,
    SOURCE_VIDEO_LOCK_WAIT_SECONDS,
)
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.media import probe_video_size
from utils.redis_client import get_redis_connection
from utils.supabase_client import supabase

_VIDEO_RAW_TTL_HOURS = 24
_WAIT_STAGE_NOTIFY_INTERVAL_SECONDS = 5.0
_WAIT_POLL_INTERVAL_SECONDS = 1.0
_UNSET_SOURCE_MAX_HEIGHT = object()
_RAW_VIDEO_STORAGE_BUCKET = "raw-videos"
_RAW_VIDEO_UPLOAD_OPTIONS = {
    "content-type": "video/mp4",
    "x-upsert": "true",
}


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


def build_raw_video_storage_path(video_id: str) -> str:
    return f"raw/{video_id}.mp4"


def _raw_cache_path(video_id: str) -> str:
    base = Path(RAW_VIDEO_CACHE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{video_id}.mp4")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_raw_video_to_storage(
    *,
    video_id: str,
    local_video_path: str,
    logger,
    job_id: str,
) -> tuple[str, str]:
    """Upload canonical raw source into private storage and return path + hash."""
    storage_path = build_raw_video_storage_path(video_id)
    file_hash = _sha256_file(local_video_path)

    with open(local_video_path, "rb") as file_obj:
        supabase.storage.from_(_RAW_VIDEO_STORAGE_BUCKET).upload(
            storage_path,
            file_obj,
            file_options=_RAW_VIDEO_UPLOAD_OPTIONS,
        )

    logger.info(
        "[%s] Uploaded canonical raw source to %s/%s",
        job_id,
        _RAW_VIDEO_STORAGE_BUCKET,
        storage_path,
    )
    return storage_path, file_hash


def _resolve_storage_video_path(
    *,
    video_id: str,
    job_id: str,
    logger,
    raw_video_storage_path: str | None,
) -> tuple[str, int, int] | None:
    if not raw_video_storage_path:
        return None

    cache_path = _raw_cache_path(video_id)
    cached_existing = _resolve_existing_video_path(
        video_id=video_id,
        job_id=job_id,
        logger=logger,
        raw_video_path=cache_path,
    )
    if cached_existing is not None:
        return cached_existing

    try:
        payload = supabase.storage.from_(_RAW_VIDEO_STORAGE_BUCKET).download(raw_video_storage_path)
    except Exception as exc:
        logger.warning(
            "[%s] Could not download canonical raw source %s/%s: %s",
            job_id,
            _RAW_VIDEO_STORAGE_BUCKET,
            raw_video_storage_path,
            exc,
        )
        return None

    if payload is None:
        return None

    try:
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        else:
            data = payload.read()
        tmp_path = f"{cache_path}.tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        logger.warning(
            "[%s] Failed to materialize canonical raw source to local cache for %s: %s",
            job_id,
            video_id,
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
    baseline_raw_video_path = str(initial_raw_video_path).strip() if initial_raw_video_path else None
    baseline_raw_storage_path = (
        str(initial_raw_video_storage_path).strip()
        if initial_raw_video_storage_path
        else None
    )
    if not prefer_fresh_download:
        existing = _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=initial_raw_video_path,
        )
        if existing is not None:
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

        storage_existing = _resolve_storage_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_storage_path=baseline_raw_storage_path,
        )
        if storage_existing is not None:
            path, width, height = storage_existing
            return SourceVideoResolution(
                video_path=path,
                width=width,
                height=height,
                strategy="downloaded_from_storage",
                wait_seconds=0.0,
                download_seconds=0.0,
                download_metadata=None,
                storage_path=baseline_raw_storage_path,
            )

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
        latest_path = row.get("raw_video_path")
        latest_url = row.get("url")
        latest_storage_path = row.get("raw_video_storage_path")
        return (
            str(latest_path).strip() if latest_path else None,
            str(latest_url).strip() if latest_url else None,
            str(latest_storage_path).strip() if latest_storage_path else None,
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
            if not baseline_raw_video_path and latest_path:
                baseline_raw_video_path = latest_path
            if not baseline_raw_storage_path and latest_storage_path:
                baseline_raw_storage_path = latest_storage_path
            latest_existing = _resolve_existing_video_path(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                raw_video_path=latest_path,
            )
            if latest_existing is not None:
                if not prefer_fresh_download or (
                    baseline_raw_video_path is None or latest_path != baseline_raw_video_path
                ):
                    path, width, height = latest_existing
                    return SourceVideoResolution(
                        video_path=path,
                        width=width,
                        height=height,
                        strategy="waited_and_reused",
                        wait_seconds=float(elapsed),
                        download_seconds=0.0,
                        download_metadata=None,
                    )

            latest_storage_existing = _resolve_storage_video_path(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                raw_video_storage_path=latest_storage_path,
            )
            if latest_storage_existing is not None:
                if not prefer_fresh_download or (
                    baseline_raw_storage_path is None
                    or latest_storage_path != baseline_raw_storage_path
                ):
                    path, width, height = latest_storage_existing
                    return SourceVideoResolution(
                        video_path=path,
                        width=width,
                        height=height,
                        strategy="waited_and_downloaded_from_storage",
                        wait_seconds=float(elapsed),
                        download_seconds=0.0,
                        download_metadata=None,
                        storage_path=latest_storage_path,
                    )

            if elapsed >= lock_wait_seconds:
                raise RuntimeError(
                    "Timed out waiting for source video download lock "
                    f"(video_id={video_id}, waited={lock_wait_seconds}s)."
                )

            time.sleep(_WAIT_POLL_INTERVAL_SECONDS)

        latest_path, latest_url, latest_storage_path = _load_latest_path_url_and_storage()
        waited_seconds = time.monotonic() - wait_started_at
        latest_existing = _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=latest_path,
        )
        if latest_existing is not None:
            if not prefer_fresh_download:
                path, width, height = latest_existing
                return SourceVideoResolution(
                    video_path=path,
                    width=width,
                    height=height,
                    strategy="waited_and_reused" if waited_seconds > 0 else "reused_existing",
                    wait_seconds=float(max(0.0, waited_seconds)),
                    download_seconds=0.0,
                    download_metadata=None,
                )
            if waited_for_lock and (
                baseline_raw_video_path is None or latest_path != baseline_raw_video_path
            ):
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

        latest_storage_existing = _resolve_storage_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_storage_path=latest_storage_path,
        )
        if latest_storage_existing is not None:
            if not prefer_fresh_download:
                path, width, height = latest_storage_existing
                return SourceVideoResolution(
                    video_path=path,
                    width=width,
                    height=height,
                    strategy=(
                        "waited_and_downloaded_from_storage"
                        if waited_seconds > 0
                        else "downloaded_from_storage"
                    ),
                    wait_seconds=float(max(0.0, waited_seconds)),
                    download_seconds=0.0,
                    download_metadata=None,
                    storage_path=latest_storage_path,
                )
            if waited_for_lock and (
                baseline_raw_storage_path is None
                or latest_storage_path != baseline_raw_storage_path
            ):
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

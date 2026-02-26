"""Shared source-video resolution with same-video download deduplication."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from config import SOURCE_VIDEO_LOCK_TTL_SECONDS, SOURCE_VIDEO_LOCK_WAIT_SECONDS
from services.video_downloader import VideoDownloader
from tasks.clips.helpers.media import probe_video_size
from utils.redis_client import get_redis_connection

_VIDEO_RAW_TTL_HOURS = 24
_WAIT_STAGE_NOTIFY_INTERVAL_SECONDS = 5.0
_WAIT_POLL_INTERVAL_SECONDS = 1.0


@dataclass(slots=True)
class SourceVideoResolution:
    video_path: str
    width: int
    height: int
    strategy: str
    wait_seconds: float
    download_seconds: float
    download_metadata: dict[str, Any] | None = None


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
    downloader: VideoDownloader,
    load_video_row: Callable[[str], dict[str, Any]],
    persist_raw_video_path: Callable[[str, str], None],
    logger,
    job_id: str,
    on_wait_for_download: Callable[[float], None] | None = None,
    on_download_start: Callable[[], None] | None = None,
) -> SourceVideoResolution:
    """Resolve a usable source-video path with same-video download deduplication."""
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

    wait_started_at = time.monotonic()
    lock_wait_seconds = max(1, int(SOURCE_VIDEO_LOCK_WAIT_SECONDS))
    lock_ttl_seconds = max(5, int(SOURCE_VIDEO_LOCK_TTL_SECONDS))
    lock_key = f"clipry:source-video:{video_id}"
    lock_token = f"{uuid.uuid4()}:{os.getpid()}"
    lock_acquired = False
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

    def _load_latest_path_and_url() -> tuple[str | None, str | None]:
        row = load_video_row(video_id) or {}
        latest_path = row.get("raw_video_path")
        latest_url = row.get("url")
        return (
            str(latest_path).strip() if latest_path else None,
            str(latest_url).strip() if latest_url else None,
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

            elapsed = time.monotonic() - wait_started_at
            _emit_wait_stage(float(elapsed))

            latest_path, _ = _load_latest_path_and_url()
            latest_existing = _resolve_existing_video_path(
                video_id=video_id,
                job_id=job_id,
                logger=logger,
                raw_video_path=latest_path,
            )
            if latest_existing is not None:
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

            if elapsed >= lock_wait_seconds:
                raise RuntimeError(
                    "Timed out waiting for source video download lock "
                    f"(video_id={video_id}, waited={lock_wait_seconds}s)."
                )

            time.sleep(_WAIT_POLL_INTERVAL_SECONDS)

        latest_path, latest_url = _load_latest_path_and_url()
        latest_existing = _resolve_existing_video_path(
            video_id=video_id,
            job_id=job_id,
            logger=logger,
            raw_video_path=latest_path,
        )
        if latest_existing is not None:
            path, width, height = latest_existing
            waited_seconds = time.monotonic() - wait_started_at
            return SourceVideoResolution(
                video_path=path,
                width=width,
                height=height,
                strategy="waited_and_reused" if waited_seconds > 0 else "reused_existing",
                wait_seconds=float(max(0.0, waited_seconds)),
                download_seconds=0.0,
                download_metadata=None,
            )

        resolved_url = str(source_url or latest_url or "").strip()
        if not resolved_url:
            raise RuntimeError("Missing source URL and raw_video_path for clip generation")

        if on_download_start is not None:
            on_download_start()

        download_started_at = time.monotonic()
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

        persist_raw_video_path(video_id, downloaded_path)
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
        )
    finally:
        if conn is not None and lock_acquired:
            _release_lock_if_owned(conn=conn, key=lock_key, token=lock_token)


def build_raw_video_metadata_update(raw_video_path: str) -> dict[str, str]:
    return {
        "raw_video_path": raw_video_path,
        "raw_video_expires_at": _expires_at_iso(),
    }

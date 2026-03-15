"""Media loading and probing helpers for social publishing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from services.social.base import PublicationMedia
from utils.media_storage import (
    create_signed_clip_url as create_minio_signed_clip_url,
    resolve_generated_clip_path,
)


def _clip_id_from_storage_path(storage_path: str) -> str | None:
    raw_value = str(storage_path or "").strip().replace("\\", "/")
    if not raw_value:
        return None
    stem = Path(raw_value).stem.strip()
    return stem or None


def create_signed_clip_url(storage_path: str, *, expires_in_seconds: int = 3600) -> str | None:
    clip_id = _clip_id_from_storage_path(storage_path)
    return create_minio_signed_clip_url(
        storage_path,
        clip_id=clip_id,
        expires_in_seconds=expires_in_seconds,
    )


def download_clip_to_path(storage_path: str, *, work_dir: str) -> str:
    resolved_path = resolve_generated_clip_path(storage_path)
    if resolved_path:
        return resolved_path

    raise RuntimeError(f"Could not materialize clip from MinIO: {storage_path}")


def probe_media(local_path: str) -> tuple[int | None, int | None, float | None]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            local_path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    format_info = payload.get("format") or {}
    video_stream = next(
        (stream for stream in streams if str(stream.get("codec_type")) == "video"),
        None,
    )

    width = None
    height = None
    duration = None
    if isinstance(video_stream, dict):
        try:
            width = int(video_stream.get("width")) if video_stream.get("width") is not None else None
            height = int(video_stream.get("height")) if video_stream.get("height") is not None else None
        except (TypeError, ValueError):
            width = None
            height = None
        try:
            raw_duration = video_stream.get("duration", format_info.get("duration"))
            duration = float(raw_duration) if raw_duration is not None else None
        except (TypeError, ValueError):
            duration = None

    return width, height, duration


def load_publication_media(storage_path: str, *, work_dir: str) -> PublicationMedia:
    local_path = download_clip_to_path(storage_path, work_dir=work_dir)
    width, height, duration = probe_media(local_path)
    return PublicationMedia(
        local_path=local_path,
        file_size=os.path.getsize(local_path),
        content_type="video/mp4",
        signed_url=create_signed_clip_url(storage_path),
        width=width,
        height=height,
        duration_seconds=duration,
    )

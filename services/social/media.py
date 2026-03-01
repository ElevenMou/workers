"""Media loading and probing helpers for social publishing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from services.social.base import PublicationMedia
from utils.supabase_client import supabase


def create_signed_clip_url(storage_path: str, *, expires_in_seconds: int = 3600) -> str | None:
    try:
        payload = supabase.storage.from_("generated-clips").create_signed_url(
            storage_path,
            expires_in_seconds,
        )
    except Exception:
        return None

    if isinstance(payload, dict):
        return payload.get("signedURL") or payload.get("signedUrl")

    if hasattr(payload, "get"):
        return payload.get("signedURL") or payload.get("signedUrl")

    return None


def download_clip_to_path(storage_path: str, *, work_dir: str) -> str:
    raw = supabase.storage.from_("generated-clips").download(storage_path)
    target_path = Path(work_dir) / "publication.mp4"
    target_path.write_bytes(raw)
    return str(target_path)


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

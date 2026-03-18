"""Media loading and probing helpers for social publishing."""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
from pathlib import Path

from services.clips.ffmpeg_ops import derive_delivery_from_master
from services.clips.render_profiles import build_delivery_encode_args
from services.media_profiles import clamped_source_fps, probe_media_profile
from services.social.base import PublicationMedia, SocialProviderError
from utils.media_storage import (
    GeneratedClipStorageError,
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
    try:
        resolved_path = resolve_generated_clip_path(storage_path, raise_on_error=True)
    except GeneratedClipStorageError as exc:
        storage_location = (
            f"{exc.bucket}/{exc.object_name}"
            if exc.object_name
            else str(storage_path or "").strip()
        )
        if exc.reason in {"invalid_storage_path", "missing_object"}:
            raise SocialProviderError(
                f"The clip asset is missing from storage ({storage_location}). Regenerate the clip before publishing.",
                code="clip_asset_missing",
            ) from exc

        raise SocialProviderError(
            f"The clip storage service could not load {storage_location}. Please retry publishing shortly.",
            code="clip_storage_unavailable",
            recoverable=exc.recoverable,
            provider_payload={
                "storage_path": exc.storage_path,
                "storage_bucket": exc.bucket,
                "storage_object": exc.object_name,
                "storage_reason": exc.reason,
            },
        ) from exc

    if resolved_path:
        return resolved_path

    raise SocialProviderError(
        f"The clip asset is missing from storage ({storage_path}). Regenerate the clip before publishing.",
        code="clip_asset_missing",
    )


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


def _build_publish_derivative_from_master(
    *,
    master_storage_path: str,
    publish_profile: str,
    delivery_profile: str | None,
    work_dir: str,
) -> str:
    master_local_path = download_clip_to_path(master_storage_path, work_dir=work_dir)
    source_profile = probe_media_profile(master_local_path)
    encode_args = build_delivery_encode_args(
        delivery_profile=delivery_profile or "social_auto_h264",
        source_profile=source_profile,
        profile_name=publish_profile,
    )
    target_fps = max(
        1,
        int(round(clamped_source_fps(source_profile, fallback=30.0, max_fps=60.0))),
    )
    output_path = os.path.join(work_dir, f"publish-{publish_profile}.mp4")
    derive_delivery_from_master(
        master_local_path,
        output_path,
        encode_args=encode_args,
        source_media_profile=source_profile,
        target_fps=target_fps,
    )
    return output_path


def load_publication_media(
    storage_path: str,
    *,
    work_dir: str,
    master_storage_path: str | None = None,
    publish_profile: str | None = None,
    delivery_profile: str | None = None,
) -> PublicationMedia:
    local_path = ""
    if master_storage_path and publish_profile:
        try:
            local_path = _build_publish_derivative_from_master(
                master_storage_path=master_storage_path,
                publish_profile=publish_profile,
                delivery_profile=delivery_profile,
                work_dir=work_dir,
            )
        except Exception:
            local_path = ""
    if not local_path:
        local_path = download_clip_to_path(storage_path, work_dir=work_dir)
    width, height, duration = probe_media(local_path)
    content_type = mimetypes.guess_type(local_path)[0] or "video/mp4"
    return PublicationMedia(
        local_path=local_path,
        file_size=os.path.getsize(local_path),
        content_type=content_type,
        signed_url=create_signed_clip_url(storage_path),
        width=width,
        height=height,
        duration_seconds=duration,
    )

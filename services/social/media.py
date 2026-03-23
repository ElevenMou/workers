"""Media loading and probing helpers for social publishing."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from services.social.base import PublicationMedia, SocialProviderError
from utils.media_storage import (
    GENERATED_CLIPS_BUCKET,
    GeneratedClipStorageError,
    build_worker_clip_url,
    create_signed_clip_url as create_minio_signed_clip_url,
    resolve_generated_clip_path,
)


def _clip_id_from_storage_path(storage_path: str) -> str | None:
    raw_value = str(storage_path or "").strip().replace("\\", "/")
    if not raw_value:
        return None
    stem = Path(raw_value).stem.strip()
    return stem or None


def is_publicly_reachable_url(url: str | None) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False

    scheme = str(parsed.scheme or "").strip().lower()
    hostname = str(parsed.hostname or "").strip().lower()
    if scheme not in {"http", "https"} or not hostname:
        return False
    if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or hostname.endswith(".localhost"):
        return False

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        if "." not in hostname:
            return False
        return True

    return not (
        host_ip.is_loopback
        or host_ip.is_private
        or host_ip.is_link_local
        or host_ip.is_multicast
        or host_ip.is_reserved
        or host_ip.is_unspecified
    )


def create_signed_clip_url(storage_path: str, *, expires_in_seconds: int = 3600) -> str | None:
    clip_id = _clip_id_from_storage_path(storage_path)
    worker_signed_url = (
        build_worker_clip_url(
            clip_id,
            expires_in_seconds=expires_in_seconds,
        )
        if clip_id
        else None
    )
    minio_signed_url = create_minio_signed_clip_url(
        storage_path,
        clip_id=clip_id,
        expires_in_seconds=expires_in_seconds,
    )
    if is_publicly_reachable_url(worker_signed_url):
        return worker_signed_url
    if is_publicly_reachable_url(minio_signed_url):
        return minio_signed_url
    return None


def _generated_clip_object_name(storage_path: str) -> str | None:
    raw_value = str(storage_path or "").strip().replace("\\", "/")
    if not raw_value:
        return None
    normalized = raw_value.lstrip("/")
    bucket_prefix = f"{GENERATED_CLIPS_BUCKET}/"
    if normalized.startswith(bucket_prefix):
        normalized = normalized[len(bucket_prefix) :]
    return normalized or None


def _generated_clip_storage_location(storage_path: str) -> str:
    object_name = _generated_clip_object_name(storage_path)
    if object_name:
        return f"{GENERATED_CLIPS_BUCKET}/{object_name}"
    return str(storage_path or "").strip()


def _clip_storage_unavailable_error(
    storage_path: str,
    *,
    reason: str,
    local_path: str | None = None,
    exc: Exception | None = None,
) -> SocialProviderError:
    object_name = _generated_clip_object_name(storage_path)
    provider_payload = {
        "storage_path": str(storage_path or "").strip(),
        "storage_bucket": GENERATED_CLIPS_BUCKET,
        "storage_object": object_name,
        "storage_reason": reason,
    }
    if local_path:
        provider_payload["local_path"] = local_path
    if exc is not None:
        provider_payload["storage_error"] = str(exc)
    return SocialProviderError(
        f"The clip storage service could not load {_generated_clip_storage_location(storage_path)}. Please retry publishing shortly.",
        code="clip_storage_unavailable",
        recoverable=True,
        provider_payload=provider_payload,
    )


def _publication_local_media_path(storage_path: str, *, work_dir: str) -> str:
    normalized = str(storage_path or "").strip().replace("\\", "/")
    suffix = Path(normalized).suffix.lower().strip() or ".mp4"
    stem = Path(normalized).stem.strip() or "clip"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem)
    safe_stem = safe_stem.strip("-_") or "clip"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return os.path.join(work_dir, f"{safe_stem}-{digest}{suffix}")


def _materialize_publication_local_copy(
    *,
    shared_path: str,
    storage_path: str,
    work_dir: str,
) -> str:
    local_path = _publication_local_media_path(storage_path, work_dir=work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if os.path.abspath(shared_path) == os.path.abspath(local_path):
        return local_path

    try:
        if os.path.exists(local_path):
            os.remove(local_path)
        try:
            os.link(shared_path, local_path)
        except OSError:
            shutil.copyfile(shared_path, local_path)
    except (FileNotFoundError, OSError) as exc:
        raise _clip_storage_unavailable_error(
            storage_path,
            reason="publication_local_copy_failed",
            local_path=local_path,
            exc=exc,
        ) from exc

    return local_path


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
        return _materialize_publication_local_copy(
            shared_path=resolved_path,
            storage_path=storage_path,
            work_dir=work_dir,
        )

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



def load_publication_media(
    storage_path: str,
    *,
    work_dir: str,
) -> PublicationMedia:
    local_path = download_clip_to_path(storage_path, work_dir=work_dir)
    try:
        width, height, duration = probe_media(local_path)
        file_size = os.path.getsize(local_path)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as exc:
        raise _clip_storage_unavailable_error(
            storage_path,
            reason="probe_failed",
            local_path=local_path,
            exc=exc,
        ) from exc
    content_type = mimetypes.guess_type(local_path)[0] or "video/mp4"
    return PublicationMedia(
        local_path=local_path,
        file_size=file_size,
        content_type=content_type,
        signed_url=create_signed_clip_url(storage_path),
        width=width,
        height=height,
        duration_seconds=duration,
    )

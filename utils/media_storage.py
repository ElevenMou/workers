"""MinIO-backed helpers for generated clips and raw videos."""

from __future__ import annotations

import hashlib
import hmac
import os
import posixpath
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

from config import (
    LOCAL_MEDIA_ROOT,
    MINIO_CLIPS_BUCKET,
    MINIO_RAW_VIDEOS_BUCKET,
    RAW_VIDEO_CACHE_DIR,
    WORKER_MEDIA_SIGNING_SECRET,
    WORKER_PUBLIC_BASE_URL,
)

GENERATED_CLIPS_BUCKET = MINIO_CLIPS_BUCKET
RAW_VIDEOS_BUCKET = MINIO_RAW_VIDEOS_BUCKET
_LOCAL_CLIP_CONTENT_TYPE = "video/mp4"
_MISSING_MINIO_OBJECT_CODES = {
    "NoSuchBucket",
    "NoSuchKey",
    "NoSuchObject",
    "ResourceNotFound",
}


class GeneratedClipStorageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        storage_path: str,
        bucket: str,
        object_name: str | None,
        reason: str,
        recoverable: bool,
    ) -> None:
        super().__init__(message)
        self.storage_path = storage_path
        self.bucket = bucket
        self.object_name = object_name
        self.reason = reason
        self.recoverable = recoverable


def _normalize_bucket_relative_path(storage_path: str, bucket: str) -> str:
    raw_value = str(storage_path or "").strip().replace("\\", "/")
    if not raw_value:
        raise ValueError("Storage path is required")

    normalized = raw_value.lstrip("/")
    bucket_prefix = f"{bucket}/"
    if normalized.startswith(bucket_prefix):
        normalized = normalized[len(bucket_prefix) :]

    normalized = posixpath.normpath(normalized)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError(f"Invalid storage path: {storage_path!r}")
    return normalized


def _bucket_root(bucket: str) -> str:
    root = Path(LOCAL_MEDIA_ROOT).expanduser() / bucket
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def resolve_local_bucket_path(bucket: str, storage_path: str) -> str:
    relative_path = _normalize_bucket_relative_path(storage_path, bucket)
    bucket_root = Path(_bucket_root(bucket)).resolve()
    target_path = (bucket_root / relative_path).resolve()
    if bucket_root != target_path and bucket_root not in target_path.parents:
        raise ValueError(f"Resolved path escapes bucket root: {storage_path!r}")
    return str(target_path)


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_bytes_atomically(path: str, payload: bytes) -> None:
    _ensure_parent_dir(path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".clipry-media-",
        suffix=".tmp",
        dir=str(Path(path).parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _best_effort_remove_empty_parent_dirs(root_path: str, target_path: str) -> None:
    try:
        root = Path(root_path).resolve()
        current = Path(target_path).resolve().parent
        while current != root and root in current.parents:
            current.rmdir()
            current = current.parent
    except OSError:
        return


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_signature(path: str, expires_at: int) -> str:
    payload = f"{path}\n{int(expires_at)}".encode("utf-8")
    secret = WORKER_MEDIA_SIGNING_SECRET.encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def build_signed_worker_media_url(
    relative_path: str,
    *,
    expires_in_seconds: int = 3600,
) -> str | None:
    if not WORKER_PUBLIC_BASE_URL or not WORKER_MEDIA_SIGNING_SECRET:
        return None

    path = str(relative_path or "").strip()
    if not path.startswith("/"):
        path = f"/{path}"

    expires_at = int(time.time()) + max(1, int(expires_in_seconds))
    query = urlencode(
        {
            "expires": str(expires_at),
            "sig": _create_signature(path, expires_at),
        }
    )
    return f"{WORKER_PUBLIC_BASE_URL}{path}?{query}"


def verify_signed_worker_media_request(
    *,
    relative_path: str,
    expires: int,
    signature: str,
) -> bool:
    if not WORKER_MEDIA_SIGNING_SECRET:
        return False
    if int(expires) < int(time.time()):
        return False

    path = str(relative_path or "").strip()
    if not path.startswith("/"):
        path = f"/{path}"

    expected = _create_signature(path, int(expires))
    return hmac.compare_digest(expected, str(signature or ""))


def build_worker_clip_url(
    clip_id: str,
    *,
    download: bool = False,
    expires_in_seconds: int = 3600,
) -> str | None:
    normalized_clip_id = str(clip_id or "").strip()
    if not normalized_clip_id:
        return None
    suffix = "/download" if download else ""
    return build_signed_worker_media_url(
        f"/media/clips/{normalized_clip_id}{suffix}",
        expires_in_seconds=expires_in_seconds,
    )


def resolve_generated_clip_local_path(storage_path: str) -> str:
    return resolve_local_bucket_path(GENERATED_CLIPS_BUCKET, storage_path)


def _generated_clip_object_name(storage_path: str) -> str:
    return _normalize_bucket_relative_path(storage_path, GENERATED_CLIPS_BUCKET)


def _minio_exception_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    return str(code or "").strip()


def _build_generated_clip_storage_error(
    *,
    storage_path: str,
    object_name: str | None,
    exc: Exception | None = None,
    reason: str | None = None,
) -> GeneratedClipStorageError:
    resolved_reason = reason or "storage_unavailable"
    if resolved_reason == "storage_unavailable":
        error_code = _minio_exception_code(exc) if exc is not None else ""
        if error_code in _MISSING_MINIO_OBJECT_CODES:
            resolved_reason = "missing_object"

    qualified_object = (
        f"{GENERATED_CLIPS_BUCKET}/{object_name}"
        if object_name
        else f"{GENERATED_CLIPS_BUCKET}/{storage_path}"
    )
    if resolved_reason == "invalid_storage_path":
        message = f"Clip storage path is invalid: {storage_path!r}."
        recoverable = False
    elif resolved_reason == "missing_object":
        message = (
            f"Clip asset is missing from MinIO at {qualified_object}. "
            "Regenerate the clip before publishing."
        )
        recoverable = False
    elif resolved_reason == "materialization_incomplete":
        message = f"Clip download from MinIO did not create a local file for {qualified_object}."
        recoverable = True
    else:
        message = f"Failed to access clip asset in MinIO at {qualified_object}: {exc}"
        recoverable = True

    return GeneratedClipStorageError(
        message,
        storage_path=storage_path,
        bucket=GENERATED_CLIPS_BUCKET,
        object_name=object_name,
        reason=resolved_reason,
        recoverable=recoverable,
    )


def ensure_generated_clip_available(
    storage_path: str | None,
    *,
    logger=None,
) -> None:
    if not storage_path:
        raise _build_generated_clip_storage_error(
            storage_path="",
            object_name=None,
            reason="invalid_storage_path",
        )

    try:
        local_path = resolve_generated_clip_local_path(storage_path)
        object_name = _generated_clip_object_name(storage_path)
    except ValueError as exc:
        if logger is not None:
            logger.warning("Invalid generated-clip storage path %s: %s", storage_path, exc)
        raise _build_generated_clip_storage_error(
            storage_path=str(storage_path),
            object_name=None,
            reason="invalid_storage_path",
        ) from exc

    if os.path.isfile(local_path):
        return

    try:
        from utils.minio_client import get_minio_client

        get_minio_client().stat_object(GENERATED_CLIPS_BUCKET, object_name)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to stat clip %s in MinIO: %s", storage_path, exc)
        raise _build_generated_clip_storage_error(
            storage_path=str(storage_path),
            object_name=object_name,
            exc=exc,
        ) from exc


def resolve_generated_clip_path(
    storage_path: str | None,
    *,
    logger=None,
    raise_on_error: bool = False,
) -> str | None:
    if not storage_path:
        return None

    try:
        local_path = resolve_generated_clip_local_path(storage_path)
        object_name = _generated_clip_object_name(storage_path)
    except ValueError as exc:
        if logger is not None:
            logger.warning("Invalid generated-clip storage path %s: %s", storage_path, exc)
        if raise_on_error:
            raise _build_generated_clip_storage_error(
                storage_path=str(storage_path),
                object_name=None,
                reason="invalid_storage_path",
            ) from exc
        return None

    if os.path.isfile(local_path):
        return local_path

    try:
        from utils.minio_client import get_minio_client

        _ensure_parent_dir(local_path)
        get_minio_client().fget_object(GENERATED_CLIPS_BUCKET, object_name, local_path)
    except Exception as exc:
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass
        if logger is not None:
            logger.warning("Failed to download clip %s from MinIO: %s", storage_path, exc)
        if raise_on_error:
            raise _build_generated_clip_storage_error(
                storage_path=str(storage_path),
                object_name=object_name,
                exc=exc,
            ) from exc
        return None

    if os.path.isfile(local_path):
        return local_path

    if raise_on_error:
        raise _build_generated_clip_storage_error(
            storage_path=str(storage_path),
            object_name=object_name,
            reason="materialization_incomplete",
        )
    return None


def delete_local_generated_clip(storage_path: str, *, logger=None) -> bool:
    try:
        local_path = resolve_generated_clip_local_path(storage_path)
    except ValueError as exc:
        if logger is not None:
            logger.warning("Invalid generated-clip delete path %s: %s", storage_path, exc)
        return False

    if not os.path.isfile(local_path):
        return False

    try:
        os.remove(local_path)
        _best_effort_remove_empty_parent_dirs(_bucket_root(GENERATED_CLIPS_BUCKET), local_path)
    except OSError as exc:
        if logger is not None:
            logger.warning("Failed to delete local generated clip %s: %s", local_path, exc)
        return False
    return True


def _cached_raw_video_path_for_storage_path(storage_path: str) -> str | None:
    normalized = _normalize_bucket_relative_path(storage_path, RAW_VIDEOS_BUCKET)
    stem = Path(normalized).stem.strip()
    if not stem:
        return None
    cache_root = Path(RAW_VIDEO_CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    return str(cache_root / f"{stem}.mp4")


def delete_local_raw_video(
    raw_video_path: str | None,
    *,
    raw_video_storage_path: str | None = None,
    logger=None,
) -> int:
    removed = 0
    candidates: list[str] = []

    if raw_video_path:
        candidates.append(str(raw_video_path).strip())
    if raw_video_storage_path:
        cached_path = _cached_raw_video_path_for_storage_path(raw_video_storage_path)
        if cached_path:
            candidates.append(cached_path)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            if os.path.isfile(normalized):
                os.remove(normalized)
                removed += 1
        except OSError as exc:
            if logger is not None:
                logger.warning("Failed to delete local raw video %s: %s", normalized, exc)

    return removed


def create_signed_clip_url(
    storage_path: str | None,
    *,
    clip_id: str | None = None,
    download: bool = False,
    expires_in_seconds: int = 3600,
) -> str | None:
    if not storage_path:
        return None

    from utils.minio_client import presigned_get_url

    object_name = _normalize_bucket_relative_path(storage_path, GENERATED_CLIPS_BUCKET)
    response_headers: dict[str, str] | None = None
    if download:
        filename = f'clip-{clip_id}.mp4' if clip_id else "clip.mp4"
        response_headers = {
            "response-content-disposition": f'attachment; filename="{filename}"',
        }

    return presigned_get_url(
        GENERATED_CLIPS_BUCKET,
        object_name,
        expires_seconds=expires_in_seconds,
        response_headers=response_headers,
    )


def upload_raw_video(
    *,
    video_id: str,
    local_video_path: str,
    logger,
    job_id: str,
) -> tuple[str | None, str | None]:
    from utils.minio_client import get_minio_client

    storage_path = _normalize_bucket_relative_path(f"raw/{video_id}.mp4", RAW_VIDEOS_BUCKET)
    file_hash = _sha256_file(local_video_path)
    get_minio_client().fput_object(
        RAW_VIDEOS_BUCKET,
        storage_path,
        local_video_path,
        content_type="video/mp4",
    )
    logger.info(
        "[%s] Uploaded canonical raw source to MinIO %s/%s",
        job_id,
        RAW_VIDEOS_BUCKET,
        storage_path,
    )
    return storage_path, file_hash


def store_generated_clip(
    *,
    local_clip_path: str,
    storage_path: str,
) -> int:
    from utils.minio_client import get_minio_client

    object_name = _normalize_bucket_relative_path(storage_path, GENERATED_CLIPS_BUCKET)
    result = get_minio_client().fput_object(
        GENERATED_CLIPS_BUCKET,
        object_name,
        local_clip_path,
        content_type=_LOCAL_CLIP_CONTENT_TYPE,
    )
    if hasattr(result, "size") and result.size:
        return int(result.size)
    return os.path.getsize(local_clip_path)


def delete_generated_clip(storage_path: str, *, logger=None) -> bool:
    try:
        from utils.minio_client import get_minio_client

        object_name = _normalize_bucket_relative_path(storage_path, GENERATED_CLIPS_BUCKET)
        get_minio_client().remove_object(GENERATED_CLIPS_BUCKET, object_name)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to delete MinIO clip %s: %s", storage_path, exc)
        return False


def delete_raw_video_from_storage(storage_path: str, *, logger=None) -> bool:
    try:
        from utils.minio_client import get_minio_client

        object_name = _normalize_bucket_relative_path(storage_path, RAW_VIDEOS_BUCKET)
        get_minio_client().remove_object(RAW_VIDEOS_BUCKET, object_name)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to delete MinIO raw video %s: %s", storage_path, exc)
        return False
def preferred_source_video_order() -> tuple[str, str]:
    return ("storage", "local")

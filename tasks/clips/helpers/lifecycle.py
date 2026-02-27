"""Shared lifecycle helpers for clip-generation task modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from utils.supabase_client import (
    assert_response_ok,
    supabase,
    update_job_status,
)

_VIDEO_UPLOAD_OPTIONS = {"content-type": "video/mp4", "cache-control": "3600"}
_DETAIL_VALUE_TYPES = (str, int, float, bool)


def parse_retention_days(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def asset_expires_at_iso(retention_days: int | None) -> str | None:
    if retention_days is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=retention_days)).isoformat()


def _is_duplicate_storage_error(exc: Exception) -> bool:
    payload = exc.args[0] if getattr(exc, "args", None) else None
    if isinstance(payload, dict):
        status_code = payload.get("statusCode")
        error_name = str(payload.get("error") or "").lower()
        message = str(payload.get("message") or "").lower()
        if status_code == 400 and (
            error_name == "duplicate" or "already exists" in message
        ):
            return True

    text = str(exc).lower()
    return "duplicate" in text or "already exists" in text


def upload_clip_with_replace(
    *,
    local_clip_path: str,
    storage_path: str,
    job_id: str,
    logger,
) -> None:
    with open(local_clip_path, "rb") as file_obj:
        try:
            supabase.storage.from_("generated-clips").upload(
                storage_path,
                file_obj,
                file_options=_VIDEO_UPLOAD_OPTIONS,
            )
            return
        except Exception as exc:
            if not _is_duplicate_storage_error(exc):
                raise

            logger.warning(
                "[%s] Storage path %s already exists. Replacing existing artifact.",
                job_id,
                storage_path,
            )
            supabase.storage.from_("generated-clips").remove([storage_path])
            file_obj.seek(0)
            supabase.storage.from_("generated-clips").upload(
                storage_path,
                file_obj,
                file_options=_VIDEO_UPLOAD_OPTIONS,
            )


def _sanitize_detail_params(detail_params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detail_params:
        return None

    safe_params: dict[str, Any] = {}
    for key, value in detail_params.items():
        if not isinstance(key, str) or not key:
            continue
        if value is None or isinstance(value, _DETAIL_VALUE_TYPES):
            safe_params[key] = value
            continue
        safe_params[key] = str(value)

    return safe_params or None


def build_progress_result_data(
    *,
    stage: str,
    detail_key: str | None = None,
    detail_params: dict[str, Any] | None = None,
    extra_result_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"stage": stage}
    if detail_key:
        payload["detail_key"] = detail_key

    safe_params = _sanitize_detail_params(detail_params)
    if safe_params:
        payload["detail_params"] = safe_params

    if extra_result_data:
        payload.update(extra_result_data)

    return payload


def update_clip_job_progress(
    *,
    job_id: str,
    progress: int,
    stage: str,
    detail_key: str | None = None,
    detail_params: dict[str, Any] | None = None,
    extra_result_data: dict[str, Any] | None = None,
) -> None:
    """Persist clip-generation progress and user-friendly detail metadata."""
    update_job_status(
        job_id,
        "processing",
        progress,
        result_data=build_progress_result_data(
            stage=stage,
            detail_key=detail_key,
            detail_params=detail_params,
            extra_result_data=extra_result_data,
        ),
    )


def best_effort_cleanup_uploaded_artifacts(
    *,
    job_id: str,
    clip_id: str,
    storage_path: str | None,
    logger,
) -> None:
    """Delete uploaded files and clear DB pointers after partial failure."""
    if storage_path:
        try:
            supabase.storage.from_("generated-clips").remove([storage_path])
        except Exception as exc:
            logger.warning(
                "[%s] Failed to delete uploaded clip artifact %s: %s",
                job_id,
                storage_path,
                exc,
            )

    if storage_path:
        try:
            clear_resp = (
                supabase.table("clips")
                .update(
                    {
                        "storage_path": None,
                        "thumbnail_path": None,
                        "file_size_bytes": None,
                        "asset_expires_at": None,
                        "asset_expired_at": None,
                    }
                )
                .eq("id", clip_id)
                .execute()
            )
            assert_response_ok(clear_resp, f"Failed to clear storage fields for {clip_id}")
        except Exception as exc:
            logger.warning(
                "[%s] Failed to clear clip storage pointers for %s: %s",
                job_id,
                clip_id,
                exc,
            )


__all__ = [
    "asset_expires_at_iso",
    "best_effort_cleanup_uploaded_artifacts",
    "build_progress_result_data",
    "parse_retention_days",
    "update_clip_job_progress",
    "upload_clip_with_replace",
]


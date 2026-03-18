"""Shared lifecycle helpers for clip-generation task modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from utils.media_storage import (
    delete_generated_clip,
    delete_local_generated_clip,
    store_generated_clip,
)
from utils.supabase_client import (
    assert_response_ok,
    supabase,
    update_job_status,
)

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


def upload_clip_with_replace(
    *,
    local_clip_path: str,
    storage_path: str,
    job_id: str,
    logger,
    allow_reencode: bool = True,
) -> int:
    if not allow_reencode:
        logger.info(
            "[%s] Upload re-encode fallback is disabled, but MinIO storage no longer "
            "requires size-based upload fallback handling.",
            job_id,
        )
    return store_generated_clip(
        local_clip_path=local_clip_path,
        storage_path=storage_path,
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
    delivery_storage_path: str | None = None,
    master_storage_path: str | None = None,
    logger,
) -> None:
    """Delete uploaded files and clear DB pointers after partial failure."""
    storage_paths = [
        path
        for path in (delivery_storage_path, storage_path, master_storage_path)
        if path
    ]
    seen_paths: set[str] = set()
    unique_paths: list[str] = []
    for path in storage_paths:
        normalized = str(path).strip()
        if not normalized or normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        unique_paths.append(normalized)

    for path in unique_paths:
        try:
            delete_local_generated_clip(path, logger=logger)
        except Exception as exc:
            logger.warning(
                "[%s] Failed to delete local clip artifact %s: %s",
                job_id,
                path,
                exc,
            )

    for path in unique_paths:
        try:
            delete_generated_clip(path, logger=logger)
        except Exception as exc:
            logger.warning(
                "[%s] Failed to delete MinIO clip artifact %s: %s",
                job_id,
                path,
                exc,
            )

    if unique_paths:
        try:
            clear_resp = (
                supabase.table("clips")
                .update(
                    {
                        "storage_path": None,
                        "delivery_storage_path": None,
                        "master_storage_path": None,
                        "delivery_profile": None,
                        "publish_profile_used": None,
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

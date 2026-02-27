"""Shared lifecycle helpers for clip-generation task modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

import ffmpeg

from utils.supabase_client import (
    assert_response_ok,
    supabase,
    update_job_status,
)

_VIDEO_UPLOAD_OPTIONS = {
    "content-type": "video/mp4",
    "cache-control": "3600",
    # Avoid duplicate-object round trips when regenerating an existing clip path.
    "x-upsert": "true",
}
_DETAIL_VALUE_TYPES = (str, int, float, bool)
_BUCKET_LIMIT_UNSET = object()
_generated_clips_bucket_limit_bytes: int | None | object = _BUCKET_LIMIT_UNSET
_UPLOAD_REENCODE_STEPS: tuple[tuple[int, str, str], ...] = (
    # crf, preset, audio bitrate
    (15, "veryslow", "192k"),
    (16, "slow", "192k"),
    (17, "slow", "192k"),
    (18, "medium", "160k"),
    (20, "medium", "128k"),
)


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


def _is_payload_too_large_storage_error(exc: Exception) -> bool:
    payload = exc.args[0] if getattr(exc, "args", None) else None
    if isinstance(payload, dict):
        status_code = payload.get("statusCode")
        error_name = str(payload.get("error") or "").lower()
        message = str(payload.get("message") or "").lower()
        if status_code == 400 and (
            "payload too large" in error_name or "maximum allowed size" in message
        ):
            return True

    text = str(exc).lower()
    return "payload too large" in text or "maximum allowed size" in text


def _upload_with_duplicate_replace(
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


def _reencode_clip_for_upload(
    *,
    input_path: str,
    output_path: str,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                vcodec="libx264",
                crf=crf,
                preset=preset,
                acodec="aac",
                audio_bitrate=audio_bitrate,
                pix_fmt="yuv420p",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg size-optimization failed: {stderr}") from exc


def _get_generated_clips_bucket_limit_bytes(*, job_id: str, logger) -> int | None:
    global _generated_clips_bucket_limit_bytes
    if _generated_clips_bucket_limit_bytes is not _BUCKET_LIMIT_UNSET:
        return _generated_clips_bucket_limit_bytes

    try:
        bucket = supabase.storage.get_bucket("generated-clips")
        raw_limit = getattr(bucket, "file_size_limit", None)
        parsed_limit = int(raw_limit) if raw_limit else None
        _generated_clips_bucket_limit_bytes = parsed_limit
        if parsed_limit:
            logger.info(
                "[%s] Detected generated-clips bucket size limit: %d bytes",
                job_id,
                parsed_limit,
            )
        else:
            logger.info(
                "[%s] generated-clips bucket has no file-size limit configured",
                job_id,
            )
        return parsed_limit
    except Exception as exc:
        logger.warning(
            "[%s] Could not read generated-clips bucket size limit: %s",
            job_id,
            exc,
        )
        _generated_clips_bucket_limit_bytes = None
        return None


def upload_clip_with_replace(
    *,
    local_clip_path: str,
    storage_path: str,
    job_id: str,
    logger,
    allow_reencode: bool = True,
) -> int:
    source_path = str(local_clip_path)
    candidate_path = source_path
    original_size = os.path.getsize(candidate_path)
    bucket_limit_bytes = _get_generated_clips_bucket_limit_bytes(job_id=job_id, logger=logger)
    attempt = 0

    # If we know the bucket cap and the original file is already above it,
    # skip the guaranteed first upload failure and optimize first.
    if bucket_limit_bytes and original_size > bucket_limit_bytes:
        if not allow_reencode:
            raise RuntimeError(
                "Generated clip exceeds storage upload size limit "
                f"(size={original_size} bytes, limit={bucket_limit_bytes} bytes) "
                "and quality-preserving mode disables upload re-encoding. "
                "Increase generated-clips bucket file-size limit."
            )

        logger.warning(
            "[%s] Original clip exceeds bucket limit (size=%d limit=%d). "
            "Applying size optimization before upload.",
            job_id,
            original_size,
            bucket_limit_bytes,
        )
        step_crf, step_preset, step_audio_bitrate = _UPLOAD_REENCODE_STEPS[attempt]
        output_path = str(
            Path(local_clip_path).with_name(
                f"{Path(local_clip_path).stem}.upload_opt_{attempt + 1}.mp4"
            )
        )
        _reencode_clip_for_upload(
            # Always encode from original source for best retained quality.
            input_path=source_path,
            output_path=output_path,
            crf=step_crf,
            preset=step_preset,
            audio_bitrate=step_audio_bitrate,
        )
        optimized_size = os.path.getsize(output_path)
        logger.warning(
            "[%s] Pre-upload optimization applied (attempt=%d crf=%d preset=%s audio=%s size=%d->%d bytes)",
            job_id,
            attempt + 1,
            step_crf,
            step_preset,
            step_audio_bitrate,
            original_size,
            optimized_size,
        )
        candidate_path = output_path
        attempt += 1

    while True:
        try:
            _upload_with_duplicate_replace(
                local_clip_path=candidate_path,
                storage_path=storage_path,
                job_id=job_id,
                logger=logger,
            )
            final_size = os.path.getsize(candidate_path)
            if attempt > 0:
                logger.warning(
                    "[%s] Upload succeeded after size optimization (%d -> %d bytes, attempts=%d)",
                    job_id,
                    original_size,
                    final_size,
                    attempt,
                )
            return final_size
        except Exception as exc:
            if not _is_payload_too_large_storage_error(exc):
                raise

            if not allow_reencode:
                current_size = os.path.getsize(candidate_path)
                limit_suffix = (
                    f" limit={bucket_limit_bytes} bytes"
                    if isinstance(bucket_limit_bytes, int) and bucket_limit_bytes > 0
                    else ""
                )
                raise RuntimeError(
                    "Generated clip exceeds storage upload size limit during upload "
                    f"(size={current_size} bytes{limit_suffix}) and quality-preserving mode "
                    "disables fallback re-encoding. Increase generated-clips bucket file-size limit."
                ) from exc

            if attempt >= len(_UPLOAD_REENCODE_STEPS):
                raise RuntimeError(
                    "Generated clip exceeds storage upload size limit after optimization attempts. "
                    "Increase storage bucket size limit or reduce output quality/duration."
                ) from exc

            if attempt == 0:
                logger.warning(
                    "[%s] Original clip is too large for storage bucket (size=%d bytes). "
                    "To preserve full quality, increase generated-clips bucket file-size limit.",
                    job_id,
                    os.path.getsize(source_path),
                )

            step_crf, step_preset, step_audio_bitrate = _UPLOAD_REENCODE_STEPS[attempt]
            output_path = str(
                Path(local_clip_path).with_name(
                    f"{Path(local_clip_path).stem}.upload_opt_{attempt + 1}.mp4"
                )
            )
            _reencode_clip_for_upload(
                # Always encode from the original render to avoid cumulative
                # quality loss across multiple fallback attempts.
                input_path=source_path,
                output_path=output_path,
                crf=step_crf,
                preset=step_preset,
                audio_bitrate=step_audio_bitrate,
            )
            optimized_size = os.path.getsize(output_path)
            logger.warning(
                "[%s] Upload payload too large. Retrying with smaller encode (attempt=%d crf=%d preset=%s audio=%s size=%d->%d bytes)",
                job_id,
                attempt + 1,
                step_crf,
                step_preset,
                step_audio_bitrate,
                os.path.getsize(candidate_path),
                optimized_size,
            )
            candidate_path = output_path
            attempt += 1


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

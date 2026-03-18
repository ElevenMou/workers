"""Signed media delivery and internal cleanup endpoints."""

from __future__ import annotations

import mimetypes
import os
import secrets
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from api_app.models import (
    DeleteClipStorageRequest,
    DeleteStorageResponse,
    DeleteVideoStorageRequest,
)
from config import WORKER_INTERNAL_API_TOKEN
from utils.media_storage import (
    GeneratedClipStorageError,
    delete_local_generated_clip,
    delete_local_raw_video,
    resolve_generated_clip_path,
    verify_signed_worker_media_request,
)
from utils.supabase_client import assert_response_ok, supabase

router = APIRouter()
_STREAM_CHUNK_BYTES = 1024 * 1024


def _require_internal_token(internal_token: str | None = Header(default=None, alias="x-worker-internal-token")) -> None:
    if not WORKER_INTERNAL_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal storage cleanup is not configured.",
        )
    if not internal_token or not secrets.compare_digest(
        str(internal_token),
        WORKER_INTERNAL_API_TOKEN,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token.",
        )


def _load_clip_storage_path(clip_id: str) -> str:
    response = (
        supabase.table("clips")
        .select("id,storage_path,delivery_storage_path,status")
        .eq("id", clip_id)
        .limit(1)
        .execute()
    )
    assert_response_ok(response, f"Failed to load clip {clip_id}")
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found.")

    clip = rows[0]
    storage_path = str(
        clip.get("delivery_storage_path")
        or clip.get("storage_path")
        or ""
    ).strip()
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip asset is not available.",
        )
    return storage_path


def _parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    if not range_header:
        return None

    value = str(range_header).strip()
    if not value.startswith("bytes=") or "," in value:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Invalid range request.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    start_str, end_str = value[6:].split("-", 1)
    if not start_str and not end_str:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Invalid range request.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    try:
        if not start_str:
            length = int(end_str)
            if length <= 0:
                raise ValueError
            start = max(file_size - length, 0)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Invalid range request.",
            headers={"Content-Range": f"bytes */{file_size}"},
        ) from exc

    if start < 0 or end < start or start >= file_size:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Requested range is not satisfiable.",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    return start, end


def _iter_file_range(path: str, start: int, end: int) -> Iterator[bytes]:
    remaining = max(0, end - start + 1)
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _build_streaming_response(
    *,
    path: str,
    request: Request,
    filename: str | None = None,
) -> Response:
    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")
    content_type, _ = mimetypes.guess_type(path)
    media_type = content_type or "application/octet-stream"
    parsed_range = _parse_range_header(range_header, file_size)

    headers = {"Accept-Ranges": "bytes"}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    if parsed_range is None:
        headers["Content-Length"] = str(file_size)
        return StreamingResponse(
            _iter_file_range(path, 0, file_size - 1),
            status_code=status.HTTP_200_OK,
            media_type=media_type,
            headers=headers,
        )

    start, end = parsed_range
    content_length = end - start + 1
    headers["Content-Length"] = str(content_length)
    headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers=headers,
    )


def _verify_signed_media_access(*, relative_path: str, expires: int, signature: str) -> None:
    if verify_signed_worker_media_request(
        relative_path=relative_path,
        expires=expires,
        signature=signature,
    ):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired media signature.",
    )


def _resolve_clip_media_path(clip_id: str) -> str:
    storage_path = _load_clip_storage_path(clip_id)

    try:
        resolved_path = resolve_generated_clip_path(storage_path, raise_on_error=True)
    except GeneratedClipStorageError as exc:
        if exc.reason in {"invalid_storage_path", "missing_object"}:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clip file could not be resolved.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clip storage is temporarily unavailable.",
        ) from exc

    if not resolved_path or not os.path.isfile(resolved_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip file could not be resolved.",
        )
    return resolved_path


@router.get("/media/clips/{clip_id}")
def stream_clip_media(
    clip_id: str,
    request: Request,
    expires: int = Query(..., ge=1),
    sig: str = Query(..., min_length=1),
):
    _verify_signed_media_access(
        relative_path=f"/media/clips/{clip_id}",
        expires=expires,
        signature=sig,
    )
    resolved_path = _resolve_clip_media_path(clip_id)
    return _build_streaming_response(path=resolved_path, request=request)


@router.get("/media/clips/{clip_id}/download")
def download_clip_media(
    clip_id: str,
    request: Request,
    expires: int = Query(..., ge=1),
    sig: str = Query(..., min_length=1),
):
    _verify_signed_media_access(
        relative_path=f"/media/clips/{clip_id}/download",
        expires=expires,
        signature=sig,
    )
    resolved_path = _resolve_clip_media_path(clip_id)
    return _build_streaming_response(
        path=resolved_path,
        request=request,
        filename=f"clip-{clip_id}.mp4",
    )


@router.post("/internal/storage/delete-clips", response_model=DeleteStorageResponse)
def delete_clip_storage(
    payload: DeleteClipStorageRequest,
    internal_token: str | None = Header(default=None, alias="x-worker-internal-token"),
):
    _require_internal_token(internal_token)
    removed = 0
    for storage_path in payload.storagePaths:
        if delete_local_generated_clip(storage_path):
            removed += 1
    return DeleteStorageResponse(requested=len(payload.storagePaths), removed=removed)


@router.post("/internal/storage/delete-videos", response_model=DeleteStorageResponse)
def delete_video_storage(
    payload: DeleteVideoStorageRequest,
    internal_token: str | None = Header(default=None, alias="x-worker-internal-token"),
):
    _require_internal_token(internal_token)
    requested = max(len(payload.rawVideoPaths), len(payload.rawVideoStoragePaths))
    removed = 0

    max_len = max(len(payload.rawVideoPaths), len(payload.rawVideoStoragePaths), 0)
    for index in range(max_len):
        raw_path = payload.rawVideoPaths[index] if index < len(payload.rawVideoPaths) else None
        raw_storage_path = (
            payload.rawVideoStoragePaths[index]
            if index < len(payload.rawVideoStoragePaths)
            else None
        )
        removed += delete_local_raw_video(
            raw_path,
            raw_video_storage_path=raw_storage_path,
        )

    return DeleteStorageResponse(requested=requested, removed=removed)

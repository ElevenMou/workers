"""Cleanup helpers for expired generated clip assets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.supabase_client import assert_response_ok, supabase

logger = logging.getLogger(__name__)


def _unique_paths(storage_path: str | None, thumbnail_path: str | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in (storage_path, thumbnail_path):
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def cleanup_expired_clip_assets(batch_size: int = 100) -> dict[str, int]:
    """Delete expired generated clip files and clear DB pointers."""
    now_iso = datetime.now(timezone.utc).isoformat()

    response = (
        supabase.table("clips")
        .select("id,storage_path,thumbnail_path,asset_expires_at,asset_expired_at")
        .lte("asset_expires_at", now_iso)
        .limit(batch_size)
        .execute()
    )
    assert_response_ok(response, "Failed to query expired clip assets")

    all_rows = list(response.data or [])
    rows = [row for row in all_rows if not row.get("asset_expired_at")]

    removed_files = 0
    expired_rows = 0

    for row in rows:
        clip_id = row.get("id")
        storage_path = row.get("storage_path")
        thumbnail_path = row.get("thumbnail_path")
        paths = _unique_paths(storage_path, thumbnail_path)

        try:
            if paths:
                supabase.storage.from_("generated-clips").remove(paths)
                removed_files += len(paths)
        except Exception as exc:
            logger.warning(
                "Failed to delete expired clip artifacts for %s (%s): %s",
                clip_id,
                paths,
                exc,
            )
            continue

        try:
            update_resp = (
                supabase.table("clips")
                .update(
                    {
                        "storage_path": None,
                        "thumbnail_path": None,
                        "file_size_bytes": None,
                        "asset_expires_at": None,
                        "asset_expired_at": now_iso,
                    }
                )
                .eq("id", clip_id)
                .execute()
            )
            assert_response_ok(
                update_resp,
                f"Failed to clear expired clip asset metadata for {clip_id}",
            )
            expired_rows += 1
        except Exception as exc:
            logger.warning(
                "Failed to clear expired clip metadata for %s: %s",
                clip_id,
                exc,
            )

    if rows:
        logger.info(
            "Expired clip-asset cleanup done: scanned=%d expired=%d removed_files=%d",
            len(rows),
            expired_rows,
            removed_files,
        )

    return {
        "scanned": len(rows),
        "expired": expired_rows,
        "removed_files": removed_files,
    }

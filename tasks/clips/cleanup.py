"""Cleanup helpers for expired generated clip assets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.media_storage import delete_generated_clip, delete_local_generated_clip
from utils.supabase_client import assert_response_ok, supabase

logger = logging.getLogger(__name__)


def cleanup_expired_clip_assets(batch_size: int = 100) -> dict[str, int]:
    """Delete expired generated clip files and clear DB pointers."""
    now_iso = datetime.now(timezone.utc).isoformat()

    response = (
        supabase.table("clips")
        .select("id,storage_path,delivery_storage_path,master_storage_path,asset_expires_at")
        .lte("asset_expires_at", now_iso)
        .is_("asset_expired_at", "null")
        .limit(batch_size)
        .execute()
    )
    assert_response_ok(response, "Failed to query expired clip assets")

    rows = list(response.data or [])

    removed_files = 0
    expired_rows = 0

    for row in rows:
        clip_id = row.get("id")
        storage_path = row.get("storage_path")
        delivery_storage_path = row.get("delivery_storage_path")
        master_storage_path = row.get("master_storage_path")
        paths = [
            path
            for path in (delivery_storage_path, storage_path, master_storage_path)
            if path
        ]
        unique_paths: list[str] = []
        seen_paths: set[str] = set()
        for path in paths:
            normalized = str(path).strip()
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            unique_paths.append(normalized)

        try:
            if unique_paths:
                for path in unique_paths:
                    delete_local_generated_clip(path, logger=logger)
                    delete_generated_clip(path, logger=logger)
                removed_files += len(unique_paths)
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
                        "delivery_storage_path": None,
                        "master_storage_path": None,
                        "delivery_profile": None,
                        "publish_profile_used": None,
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

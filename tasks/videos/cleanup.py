"""Cleanup helpers for expired local raw video files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.media_storage import delete_local_raw_video, delete_raw_video_from_storage
from utils.supabase_client import assert_response_ok, supabase

logger = logging.getLogger(__name__)


def cleanup_expired_raw_videos(batch_size: int = 100) -> dict[str, int]:
    """Delete expired raw video files and clear DB pointers."""
    now_iso = datetime.now(timezone.utc).isoformat()
    response = (
        supabase.table("videos")
        .select("id,raw_video_path,raw_video_storage_path,raw_video_expires_at")
        .lte("raw_video_expires_at", now_iso)
        .limit(batch_size)
        .execute()
    )
    assert_response_ok(response, "Failed to query expired raw videos")

    rows = list(response.data or [])
    removed = 0
    cleared = 0

    for row in rows:
        video_id = row.get("id")
        raw_video_path = row.get("raw_video_path")
        raw_video_storage_path = row.get("raw_video_storage_path")

        path_removed = delete_local_raw_video(
            raw_video_path,
            raw_video_storage_path=raw_video_storage_path,
            logger=logger,
        )
        removed += path_removed
        path_cleared = path_removed > 0 or not raw_video_path

        storage_cleared = False
        if raw_video_storage_path:
            try:
                storage_cleared = delete_raw_video_from_storage(
                    raw_video_storage_path,
                    logger=logger,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to delete expired canonical raw video %s for %s: %s",
                    raw_video_storage_path,
                    video_id,
                    exc,
                )
        else:
            storage_cleared = True

        if not path_cleared or not storage_cleared:
            continue

        try:
            clear_resp = (
                supabase.table("videos")
                .update(
                    {
                        "raw_video_path": None,
                        "raw_video_storage_path": None,
                        "raw_video_storage_etag": None,
                        "raw_video_expires_at": None,
                    }
                )
                .eq("id", video_id)
                .execute()
            )
            assert_response_ok(clear_resp, f"Failed to clear raw-video fields for {video_id}")
            cleared += 1
        except Exception as exc:
            logger.warning(
                "Failed to clear expired raw-video metadata for %s: %s",
                video_id,
                exc,
            )

    if rows:
        logger.info(
            "Expired raw-video cleanup done: scanned=%d removed=%d cleared=%d",
            len(rows),
            removed,
            cleared,
        )

    return {"scanned": len(rows), "removed": removed, "cleared": cleared}


"""Startup recovery and cleanup helpers for supervisor."""

import os
from datetime import datetime, timedelta, timezone

from worker_supervisor.state import logger


def env_bool(name: str, default: bool) -> bool:
    """Parse environment variable to bool."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on"}


def list_queue_job_ids(queue) -> list[str]:
    """Return queued job ids for an RQ Queue object."""
    job_ids = getattr(queue, "job_ids", None)
    if callable(job_ids):
        return list(job_ids())
    if job_ids is not None:
        return list(job_ids)

    get_job_ids = getattr(queue, "get_job_ids", None)
    if callable(get_job_ids):
        return list(get_job_ids())

    return []


def mark_jobs_failed(job_ids: list[str], reason: str):
    """Best-effort: mark dropped queue jobs as failed in Supabase."""
    if not job_ids:
        return

    try:
        from utils.supabase_client import assert_response_ok, supabase  # noqa: E402
    except Exception as exc:
        logger.warning("Could not import Supabase client to mark dropped jobs: %s", exc)
        return

    payload = {
        "status": "failed",
        "error_message": reason,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    updated = 0
    for job_id in job_ids:
        try:
            resp = supabase.table("jobs").update(payload).eq("id", job_id).execute()
            assert_response_ok(resp, f"Failed to mark dropped job {job_id} as failed")
            updated += 1
        except Exception as exc:
            logger.warning("Failed to update dropped job %s: %s", job_id, exc)
        finally:
            try:
                from utils.redis_client import release_job_admission

                release_job_admission(job_id)
            except Exception as exc:
                logger.warning(
                    "Failed to release admission token while failing job %s: %s",
                    job_id,
                    exc,
                )

    logger.info("Marked %d/%d dropped jobs as failed", updated, len(job_ids))


def purge_startup_backlog(conn, queue_names: list[str]):
    """Clear queued jobs on startup so old backlog is not processed."""
    if not env_bool("PURGE_QUEUED_JOBS_ON_START", False):
        logger.info("Startup queue purge disabled (PURGE_QUEUED_JOBS_ON_START=false)")
        return

    try:
        from utils.redis_client import get_queue  # noqa: E402
    except Exception as exc:
        logger.warning("Could not import queue helper for startup purge: %s", exc)
        return

    total_dropped = 0
    for queue_name in queue_names:
        try:
            q = get_queue(queue_name, conn)
            job_ids = list_queue_job_ids(q)
            if not job_ids:
                continue

            q.empty()
            dropped = len(job_ids)
            total_dropped += dropped
            logger.warning(
                "Dropped %d queued jobs from '%s' at startup",
                dropped,
                queue_name,
            )
            mark_jobs_failed(
                job_ids,
                "Dropped from Redis queue during worker startup purge",
            )
        except Exception as exc:
            logger.warning("Failed startup queue purge for '%s': %s", queue_name, exc)

    if total_dropped == 0:
        logger.info("Startup queue purge complete: no queued jobs found")
    else:
        logger.info("Startup queue purge complete: dropped %d queued jobs", total_dropped)


def cleanup_started_jobs_on_start(conn, queue_names: list[str]):
    """Recover interrupted jobs left in Started registry after restarts."""
    if not env_bool("FAIL_STARTED_JOBS_ON_START", True):
        logger.info(
            "Startup started-job cleanup disabled (FAIL_STARTED_JOBS_ON_START=false)"
        )
        return

    try:
        from rq.job import Job
        from rq.registry import StartedJobRegistry
        from utils.redis_client import get_queue  # noqa: E402
    except Exception as exc:
        logger.warning("Could not import started-job cleanup helpers: %s", exc)
        return

    recovered_job_ids: list[str] = []
    for queue_name in queue_names:
        try:
            q = get_queue(queue_name, conn)
            reg = StartedJobRegistry(queue=q)
            started_ids = reg.get_job_ids()
            for job_id in started_ids:
                try:
                    reg.remove(job_id, delete_job=False)
                except Exception:
                    pass

                try:
                    job = Job.fetch(job_id, connection=conn)
                    job.set_status("failed")
                except Exception:
                    pass

                recovered_job_ids.append(job_id)
        except Exception as exc:
            logger.warning("Failed started-job cleanup for '%s': %s", queue_name, exc)

    if recovered_job_ids:
        logger.warning(
            "Recovered %d interrupted started jobs on startup",
            len(recovered_job_ids),
        )
        mark_jobs_failed(
            recovered_job_ids,
            "Marked failed: interrupted job recovered on worker startup",
        )
    else:
        logger.info("Startup started-job cleanup complete: no interrupted jobs found")


def recover_processing_rows_on_start():
    """Mark orphaned DB rows in processing state as failed on worker startup."""
    if not env_bool("FAIL_PROCESSING_ROWS_ON_START", True):
        logger.info(
            "Startup DB processing-row recovery disabled "
            "(FAIL_PROCESSING_ROWS_ON_START=false)"
        )
        return

    try:
        from utils.supabase_client import assert_response_ok, supabase  # noqa: E402
    except Exception as exc:
        logger.warning(
            "Could not import Supabase client for processing-row recovery: %s",
            exc,
        )
        return

    jobs_resp = (
        supabase.table("jobs")
        .select("id,type,clip_id,video_id")
        .eq("status", "processing")
        .execute()
    )
    assert_response_ok(jobs_resp, "Failed to query processing jobs on startup")

    rows = list(jobs_resp.data or [])
    if not rows:
        logger.info("Startup DB processing-row recovery: no processing jobs found")
        return

    reason = "Marked failed: interrupted during worker restart"
    job_ids = [row["id"] for row in rows if row.get("id")]
    mark_jobs_failed(job_ids, reason)

    for row in rows:
        clip_id = row.get("clip_id")
        video_id = row.get("video_id")
        job_type = row.get("type")

        if clip_id:
            try:
                clip_resp = (
                    supabase.table("clips")
                    .update({"status": "failed", "error_message": reason})
                    .eq("id", clip_id)
                    .execute()
                )
                assert_response_ok(clip_resp, f"Failed to mark clip {clip_id} failed")
            except Exception as exc:
                logger.warning("Failed to recover clip %s: %s", clip_id, exc)

        if job_type == "analyze_video" and video_id:
            try:
                video_resp = (
                    supabase.table("videos")
                    .update({"status": "failed", "error_message": reason})
                    .eq("id", video_id)
                    .execute()
                )
                assert_response_ok(video_resp, f"Failed to mark video {video_id} failed")
            except Exception as exc:
                logger.warning("Failed to recover video %s: %s", video_id, exc)

    logger.warning("Startup DB processing-row recovery marked %d jobs as failed", len(rows))


def collect_started_job_ids(conn, queue_names: list[str]) -> set[str]:
    """Collect in-flight RQ started-job ids across configured queues."""
    try:
        from rq.registry import StartedJobRegistry
        from utils.redis_client import get_queue  # noqa: E402
    except Exception as exc:
        logger.warning("Could not import started-job registry helpers: %s", exc)
        return set()

    started_job_ids: set[str] = set()
    for queue_name in queue_names:
        try:
            queue = get_queue(queue_name, conn)
            registry = StartedJobRegistry(queue=queue)
            started_job_ids.update(str(job_id) for job_id in registry.get_job_ids())
        except Exception as exc:
            logger.warning(
                "Failed reading Started registry for '%s' during stale sweep: %s",
                queue_name,
                exc,
            )
    return started_job_ids


def recover_stale_processing_rows(
    *,
    conn,
    queue_names: list[str],
    stale_seconds: int,
) -> int:
    """
    Mark stale processing DB rows as failed when they are not in active Started registries.

    Returns the number of jobs transitioned to failed.
    """
    stale_seconds = int(stale_seconds)
    if stale_seconds <= 0:
        return 0

    try:
        from utils.supabase_client import assert_response_ok, supabase  # noqa: E402
    except Exception as exc:
        logger.warning("Could not import Supabase client for stale sweep: %s", exc)
        return 0

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()

    _PAGE_SIZE = 500

    def _paginated_query(query_builder, context_msg: str) -> list[dict]:
        """Fetch all matching rows in pages of _PAGE_SIZE."""
        all_rows: list[dict] = []
        offset = 0
        while True:
            page_resp = (
                query_builder
                .range(offset, offset + _PAGE_SIZE - 1)
                .execute()
            )
            assert_response_ok(page_resp, context_msg)
            page = list(page_resp.data or [])
            all_rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return all_rows

    rows = _paginated_query(
        supabase.table("jobs")
        .select("id,type,clip_id,video_id,started_at,created_at")
        .eq("status", "processing")
        .lt("started_at", cutoff_iso),
        "Failed to query stale processing jobs by started_at",
    )

    seen_ids = {row.get("id") for row in rows}
    created_rows = _paginated_query(
        supabase.table("jobs")
        .select("id,type,clip_id,video_id,started_at,created_at")
        .eq("status", "processing")
        .is_("started_at", "null")
        .lt("created_at", cutoff_iso),
        "Failed to query stale processing jobs by created_at fallback",
    )
    for row in created_rows:
        row_id = row.get("id")
        if row_id and row_id not in seen_ids:
            rows.append(row)
            seen_ids.add(row_id)

    if not rows:
        return 0

    active_started_job_ids = collect_started_job_ids(conn, queue_names)
    stale_rows = [row for row in rows if str(row.get("id") or "") not in active_started_job_ids]
    if not stale_rows:
        return 0

    reason = (
        "Marked failed: stale processing heartbeat exceeded "
        f"{stale_seconds}s and job was not active in worker started registries"
    )
    stale_job_ids = [str(row["id"]) for row in stale_rows if row.get("id")]
    mark_jobs_failed(stale_job_ids, reason)

    clip_failures = 0
    video_failures = 0
    for row in stale_rows:
        clip_id = row.get("clip_id")
        video_id = row.get("video_id")
        job_type = str(row.get("type") or "")

        if clip_id:
            try:
                clip_resp = (
                    supabase.table("clips")
                    .update({"status": "failed", "error_message": reason})
                    .eq("id", clip_id)
                    .execute()
                )
                assert_response_ok(clip_resp, f"Failed to mark clip {clip_id} failed")
                clip_failures += 1
            except Exception as exc:
                logger.warning("Failed stale-recovery update for clip %s: %s", clip_id, exc)

        if video_id and job_type in {"analyze_video", "custom_clip"}:
            try:
                video_resp = (
                    supabase.table("videos")
                    .update({"status": "failed", "error_message": reason})
                    .eq("id", video_id)
                    .execute()
                )
                assert_response_ok(video_resp, f"Failed to mark video {video_id} failed")
                video_failures += 1
            except Exception as exc:
                logger.warning("Failed stale-recovery update for video %s: %s", video_id, exc)

    logger.warning(
        "Runtime stale-processing recovery marked %d jobs failed (clips=%d videos=%d cutoff=%ss)",
        len(stale_rows),
        clip_failures,
        video_failures,
        stale_seconds,
    )
    try:
        conn.incrby("clipry:metrics:stale_recoveries:total", len(stale_rows))
        conn.incrby("clipry:metrics:stale_recoveries:clips", int(clip_failures))
        conn.incrby("clipry:metrics:stale_recoveries:videos", int(video_failures))
    except Exception as exc:
        logger.warning("Failed to increment stale-recovery metrics: %s", exc)
    return len(stale_rows)

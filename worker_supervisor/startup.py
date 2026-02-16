"""Startup recovery and cleanup helpers for supervisor."""

import os
from datetime import datetime, timezone

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

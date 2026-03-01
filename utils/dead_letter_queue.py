"""Dead letter queue for permanently failed jobs.

Jobs that exhaust all RQ retries are recorded here for inspection,
alerting, and potential reprocessing.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DLQ_KEY = "clipry:dlq:jobs"
DLQ_COUNTER_KEY = "clipry:metrics:dlq:total"
DLQ_MAX_LENGTH = 10_000
DLQ_ENTRY_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def record_dead_letter(
    conn,
    *,
    job_id: str,
    queue_name: str,
    task_path: str,
    error_message: str,
    job_data: dict | None = None,
    attempt_count: int = 0,
) -> None:
    """Push a permanently failed job onto the dead letter list."""
    entry = {
        "job_id": job_id,
        "queue_name": queue_name,
        "task_path": task_path,
        "error_message": error_message[:2000],
        "job_data_summary": _summarize_job_data(job_data),
        "attempt_count": attempt_count,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "recorded_at_epoch": time.time(),
    }

    try:
        pipe = conn.pipeline()
        pipe.lpush(DLQ_KEY, json.dumps(entry))
        pipe.ltrim(DLQ_KEY, 0, DLQ_MAX_LENGTH - 1)
        pipe.incr(DLQ_COUNTER_KEY)
        pipe.execute()
        logger.warning(
            "Job %s moved to dead letter queue (queue=%s task=%s): %s",
            job_id,
            queue_name,
            task_path,
            error_message[:200],
        )
    except Exception as exc:
        logger.error(
            "Failed to record dead letter for job %s: %s",
            job_id,
            exc,
        )


def get_dlq_entries(
    conn,
    *,
    offset: int = 0,
    limit: int = 50,
) -> list[dict]:
    """Read recent dead letter entries."""
    try:
        raw_entries = conn.lrange(DLQ_KEY, offset, offset + limit - 1)
        return [json.loads(entry) for entry in raw_entries]
    except Exception as exc:
        logger.warning("Failed to read DLQ entries: %s", exc)
        return []


def get_dlq_count(conn) -> int:
    """Get total count of dead letter events (lifetime counter)."""
    try:
        value = conn.get(DLQ_COUNTER_KEY)
        return int(value) if value else 0
    except Exception:
        return 0


def _summarize_job_data(data: dict | None) -> dict | None:
    if not data:
        return None
    return {
        k: v
        for k, v in data.items()
        if k in {"jobId", "videoId", "clipId", "userId", "publicationId", "type"}
    }

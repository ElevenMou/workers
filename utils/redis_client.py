import logging

import redis
from rq import Queue

from config import (
    CLIP_JOB_TIMEOUT,
    MAX_CLIP_QUEUE_DEPTH,
    MAX_VIDEO_QUEUE_DEPTH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_MAX_CONNECTIONS,
    REDIS_PASSWORD,
    REDIS_PORT,
    VIDEO_JOB_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Queue configuration - maps queue name -> RQ Queue kwargs
# ---------------------------------------------------------------------------
QUEUE_CONFIG: dict[str, dict] = {
    "video-processing-priority": {"default_timeout": VIDEO_JOB_TIMEOUT},
    "video-processing": {"default_timeout": VIDEO_JOB_TIMEOUT},
    "clip-generation-priority": {"default_timeout": CLIP_JOB_TIMEOUT},
    "clip-generation": {"default_timeout": CLIP_JOB_TIMEOUT},
}

# Queue group -> (priority queue, normal queue, max depth).
_QUEUE_GROUPS: dict[str, tuple[str, str, int]] = {
    "video": ("video-processing-priority", "video-processing", MAX_VIDEO_QUEUE_DEPTH),
    "clip": ("clip-generation-priority", "clip-generation", MAX_CLIP_QUEUE_DEPTH),
}

# Map each queue name back to its group key for quick lookup.
_QUEUE_TO_GROUP: dict[str, str] = {}
for _group_key, (_pq, _nq, _) in _QUEUE_GROUPS.items():
    _QUEUE_TO_GROUP[_pq] = _group_key
    _QUEUE_TO_GROUP[_nq] = _group_key

# Legacy shared key (kept for compatibility while migrating to per-group keys).
WORKER_SCALE_KEY = "clipry:workers:desired"
WORKER_SCALE_VIDEO_KEY = "clipry:workers:desired:video"
WORKER_SCALE_CLIP_KEY = "clipry:workers:desired:clip"
_WORKER_GROUP_KEY_BY_NAME = {
    "video": WORKER_SCALE_VIDEO_KEY,
    "clip": WORKER_SCALE_CLIP_KEY,
}

QUEUE_REJECTS_TOTAL_KEY = "clipry:metrics:queue_rejects:total"
QUEUE_REJECTS_VIDEO_KEY = "clipry:metrics:queue_rejects:video"
QUEUE_REJECTS_CLIP_KEY = "clipry:metrics:queue_rejects:clip"
_QUEUE_REJECT_KEY_BY_GROUP = {
    "video": QUEUE_REJECTS_VIDEO_KEY,
    "clip": QUEUE_REJECTS_CLIP_KEY,
}

MAX_WORKER_COUNT = 256


class QueueFullError(Exception):
    """Raised when a queue exceeds its configured maximum depth."""


# ---------------------------------------------------------------------------
# Connection pool (per-process; each subprocess re-imports and gets its own)
# ---------------------------------------------------------------------------
_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    max_connections=REDIS_MAX_CONNECTIONS,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)


# ---------------------------------------------------------------------------
# Factory helpers (safe across subprocesses - each gets its own pool)
# ---------------------------------------------------------------------------
def get_redis_connection() -> redis.Redis:
    """Return a Redis client backed by the module-level connection pool."""
    return redis.Redis(connection_pool=_pool)


def get_queue(name: str, connection: redis.Redis | None = None) -> Queue:
    """Create a single :class:`Queue` with its configured timeout."""
    conn = connection or get_redis_connection()
    cfg = QUEUE_CONFIG.get(name, {})
    return Queue(name, connection=conn, default_timeout=cfg.get("default_timeout", 600))


def get_queues(names: list[str], connection: redis.Redis | None = None) -> list[Queue]:
    """Create multiple queues sharing one Redis connection."""
    conn = connection or get_redis_connection()
    return [get_queue(n, conn) for n in names]


def get_queue_depth(queue_name: str, conn: redis.Redis | None = None) -> int:
    """Return the number of jobs waiting in *queue_name*."""
    c = conn or get_redis_connection()
    return int(c.llen(f"rq:queue:{queue_name}"))


def _increment_queue_reject_counters(*, conn: redis.Redis, group_key: str) -> None:
    """Best-effort queue reject counters for operational monitoring."""
    try:
        conn.incr(QUEUE_REJECTS_TOTAL_KEY)
        group_counter_key = _QUEUE_REJECT_KEY_BY_GROUP.get(group_key)
        if group_counter_key:
            conn.incr(group_counter_key)
    except Exception as exc:
        logger.warning("Failed to increment queue reject metrics for %s: %s", group_key, exc)


def get_queue_reject_counts(conn: redis.Redis | None = None) -> dict[str, int]:
    c = conn or get_redis_connection()

    def _as_int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "total": _as_int(c.get(QUEUE_REJECTS_TOTAL_KEY)),
        "video": _as_int(c.get(QUEUE_REJECTS_VIDEO_KEY)),
        "clip": _as_int(c.get(QUEUE_REJECTS_CLIP_KEY)),
    }


def _check_backpressure(queue_name: str, conn: redis.Redis) -> None:
    """Raise :class:`QueueFullError` if the queue group is at capacity."""
    group_key = _QUEUE_TO_GROUP.get(queue_name)
    if group_key is None:
        return  # Unknown queue - skip check.

    priority_q, normal_q, max_depth = _QUEUE_GROUPS[group_key]
    total = conn.llen(f"rq:queue:{priority_q}") + conn.llen(f"rq:queue:{normal_q}")
    if total >= max_depth:
        _increment_queue_reject_counters(conn=conn, group_key=group_key)
        raise QueueFullError(
            f"Queue group '{group_key}' is full ({total}/{max_depth}). "
            "Please retry later."
        )


def enqueue_job(queue_name: str, task_path: str, job_data: dict, *, job_id: str):
    """Enqueue a job after verifying queue depth is within limits."""
    conn = get_redis_connection()
    _check_backpressure(queue_name, conn)
    queue = get_queue(queue_name, conn)
    return queue.enqueue(task_path, job_data, job_id=job_id)


def _decode_redis_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _normalize_worker_count(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, min(parsed, MAX_WORKER_COUNT))


def _get_scale_key_for_group(group: str) -> str:
    key = _WORKER_GROUP_KEY_BY_NAME.get(group)
    if key is None:
        raise ValueError(f"Unsupported worker group: {group}")
    return key


def get_group_worker_scale_target(
    *,
    group: str,
    connection: redis.Redis | None = None,
    default: int = 0,
) -> int:
    """Read desired worker count for one group, with legacy fallback."""
    conn = connection or get_redis_connection()
    group_key = _get_scale_key_for_group(group)
    group_value = _decode_redis_value(conn.get(group_key))
    if group_value is not None:
        return _normalize_worker_count(group_value, default)

    # Backward-compatible fallback to old shared hash during migration.
    data = conn.hgetall(WORKER_SCALE_KEY) or {}
    legacy_field = f"{group}_workers"
    legacy_value = _decode_redis_value(data.get(legacy_field) or data.get(legacy_field.encode()))
    return _normalize_worker_count(legacy_value, default)


def set_group_worker_scale_target(
    *,
    group: str,
    workers: int,
    connection: redis.Redis | None = None,
    default: int = 0,
) -> int:
    """Persist desired worker count for one group and mirror to legacy hash."""
    conn = connection or get_redis_connection()
    group_key = _get_scale_key_for_group(group)
    normalized = _normalize_worker_count(workers, default)
    conn.set(group_key, normalized)

    # Mirror to legacy shared hash for backward compatibility during rollout.
    conn.hset(WORKER_SCALE_KEY, mapping={f"{group}_workers": normalized})
    return normalized


def get_worker_scale_target(
    *,
    connection: redis.Redis | None = None,
    default_video: int = 0,
    default_clip: int = 0,
) -> tuple[int, int]:
    """Read desired worker counts from Redis, falling back to defaults."""
    conn = connection or get_redis_connection()
    video_workers = get_group_worker_scale_target(
        group="video",
        connection=conn,
        default=default_video,
    )
    clip_workers = get_group_worker_scale_target(
        group="clip",
        connection=conn,
        default=default_clip,
    )
    return video_workers, clip_workers


def set_worker_scale_target(
    *,
    video_workers: int | None = None,
    clip_workers: int | None = None,
    connection: redis.Redis | None = None,
    default_video: int = 0,
    default_clip: int = 0,
) -> tuple[int, int]:
    """Persist desired worker counts and return the normalized final values."""
    if video_workers is None and clip_workers is None:
        raise ValueError("At least one worker count must be provided")

    conn = connection or get_redis_connection()
    current_video, current_clip = get_worker_scale_target(
        connection=conn,
        default_video=default_video,
        default_clip=default_clip,
    )

    final_video = (
        set_group_worker_scale_target(
            group="video",
            workers=video_workers,
            connection=conn,
            default=current_video,
        )
        if video_workers is not None
        else current_video
    )
    final_clip = (
        set_group_worker_scale_target(
            group="clip",
            workers=clip_workers,
            connection=conn,
            default=current_clip,
        )
        if clip_workers is not None
        else current_clip
    )

    return final_video, final_clip


# ---------------------------------------------------------------------------
# Module-level singletons (used by api.py - single process)
# ---------------------------------------------------------------------------
redis_conn = get_redis_connection()
video_priority_queue = get_queue("video-processing-priority", redis_conn)
video_queue = get_queue("video-processing", redis_conn)
clip_priority_queue = get_queue("clip-generation-priority", redis_conn)
clip_queue = get_queue("clip-generation", redis_conn)


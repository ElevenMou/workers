import logging

import redis
from rq import Queue, Retry

from config import (
    CLIP_JOB_TIMEOUT,
    MAX_CLIP_QUEUE_DEPTH,
    MAX_SOCIAL_QUEUE_DEPTH,
    MAX_VIDEO_QUEUE_DEPTH,
    REDIS_DB,
    REDIS_HOST,
    REDIS_MAX_CONNECTIONS,
    REDIS_PASSWORD,
    REDIS_PORT,
    SOCIAL_JOB_TIMEOUT,
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
    "social-publishing-priority": {"default_timeout": SOCIAL_JOB_TIMEOUT},
    "social-publishing": {"default_timeout": SOCIAL_JOB_TIMEOUT},
}

# Queue group -> (priority queue, normal queue, max depth).
_QUEUE_GROUPS: dict[str, tuple[str, str, int]] = {
    "video": ("video-processing-priority", "video-processing", MAX_VIDEO_QUEUE_DEPTH),
    "clip": ("clip-generation-priority", "clip-generation", MAX_CLIP_QUEUE_DEPTH),
    "social": ("social-publishing-priority", "social-publishing", MAX_SOCIAL_QUEUE_DEPTH),
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
WORKER_SCALE_SOCIAL_KEY = "clipry:workers:desired:social"
_WORKER_GROUP_KEY_BY_NAME = {
    "video": WORKER_SCALE_VIDEO_KEY,
    "clip": WORKER_SCALE_CLIP_KEY,
    "social": WORKER_SCALE_SOCIAL_KEY,
}

QUEUE_REJECTS_TOTAL_KEY = "clipry:metrics:queue_rejects:total"
QUEUE_REJECTS_VIDEO_KEY = "clipry:metrics:queue_rejects:video"
QUEUE_REJECTS_CLIP_KEY = "clipry:metrics:queue_rejects:clip"
QUEUE_REJECTS_SOCIAL_KEY = "clipry:metrics:queue_rejects:social"
_QUEUE_REJECT_KEY_BY_GROUP = {
    "video": QUEUE_REJECTS_VIDEO_KEY,
    "clip": QUEUE_REJECTS_CLIP_KEY,
    "social": QUEUE_REJECTS_SOCIAL_KEY,
}
ADMISSION_VIDEO_KEY = "clipry:admission:video"
ADMISSION_CLIP_KEY = "clipry:admission:clip"
ADMISSION_SOCIAL_KEY = "clipry:admission:social"
_ADMISSION_KEY_BY_GROUP = {
    "video": ADMISSION_VIDEO_KEY,
    "clip": ADMISSION_CLIP_KEY,
    "social": ADMISSION_SOCIAL_KEY,
}
ADMISSION_RELEASED_PREFIX = "clipry:admission:released:"
ADMISSION_JOB_GROUP_PREFIX = "clipry:admission:job-group:"
ADMISSION_RELEASES_TOTAL_KEY = "clipry:metrics:admission_releases:total"
ADMISSION_RELEASES_VIDEO_KEY = "clipry:metrics:admission_releases:video"
ADMISSION_RELEASES_CLIP_KEY = "clipry:metrics:admission_releases:clip"
ADMISSION_RELEASES_SOCIAL_KEY = "clipry:metrics:admission_releases:social"
_ADMISSION_RELEASES_KEY_BY_GROUP = {
    "video": ADMISSION_RELEASES_VIDEO_KEY,
    "clip": ADMISSION_RELEASES_CLIP_KEY,
    "social": ADMISSION_RELEASES_SOCIAL_KEY,
}
_ADMISSION_RELEASED_TTL_SECONDS = 60 * 60 * 24 * 7
_ADMISSION_JOB_GROUP_TTL_SECONDS = 60 * 60 * 24 * 7
_DEFAULT_RETRY_INTERVALS_SECONDS = (30, 120, 600)
_DEFAULT_RETRY_MAX = len(_DEFAULT_RETRY_INTERVALS_SECONDS)

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


def _increment_admission_release_counters(*, conn: redis.Redis, group_key: str) -> None:
    try:
        conn.incr(ADMISSION_RELEASES_TOTAL_KEY)
        group_counter_key = _ADMISSION_RELEASES_KEY_BY_GROUP.get(group_key)
        if group_counter_key:
            conn.incr(group_counter_key)
    except Exception as exc:
        logger.warning(
            "Failed to increment admission release metrics for %s: %s",
            group_key,
            exc,
        )


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
        "social": _as_int(c.get(QUEUE_REJECTS_SOCIAL_KEY)),
    }


def _admission_key_for_group(group_key: str) -> str:
    key = _ADMISSION_KEY_BY_GROUP.get(group_key)
    if key is None:
        raise ValueError(f"Unsupported admission group: {group_key}")
    return key


def admission_group_for_queue(queue_name: str) -> str | None:
    return _QUEUE_TO_GROUP.get(queue_name)


def admission_group_for_job_type(job_type: str | None) -> str | None:
    normalized = str(job_type or "").strip().lower()
    if normalized == "analyze_video":
        return "video"
    if normalized in {"generate_clip", "custom_clip"}:
        return "clip"
    if normalized == "publish_clip":
        return "social"
    return None


def _acquire_admission_token(*, conn: redis.Redis, group_key: str, max_depth: int) -> int:
    admission_key = _admission_key_for_group(group_key)
    script = """
local key = KEYS[1]
local max_depth = tonumber(ARGV[1])
local current = tonumber(redis.call('GET', key) or '0')
if current >= max_depth then
  return -1
end
local updated = redis.call('INCR', key)
if tonumber(updated) > max_depth then
  redis.call('DECR', key)
  return -1
end
return tonumber(updated)
"""
    try:
        token = int(conn.eval(script, 1, admission_key, int(max_depth)))
    except Exception as exc:
        logger.warning(
            "Admission token acquire failed for %s (max=%d): %s",
            group_key,
            int(max_depth),
            exc,
        )
        raise

    if token < 0:
        _increment_queue_reject_counters(conn=conn, group_key=group_key)
        raise QueueFullError(
            f"Queue group '{group_key}' is full ({max_depth}/{max_depth}). "
            "Please retry later."
        )
    return token


def _rollback_admission(
    *,
    conn: redis.Redis,
    group_key: str,
    job_id: str,
) -> None:
    admission_key = _admission_key_for_group(group_key)
    script = """
local key = KEYS[1]
local current = tonumber(redis.call('GET', key) or '0')
if current > 0 then
  redis.call('DECR', key)
end
return 1
"""
    try:
        conn.eval(script, 1, admission_key)
    except Exception as exc:
        logger.warning("Failed to rollback admission token for %s: %s", job_id, exc)
    try:
        conn.delete(f"{ADMISSION_JOB_GROUP_PREFIX}{job_id}")
    except Exception as exc:
        logger.warning("Failed to clear admission job-group key for %s: %s", job_id, exc)


def release_job_admission(
    job_id: str,
    *,
    job_group: str | None = None,
    connection: redis.Redis | None = None,
) -> bool:
    """Release one admission slot for a job id, idempotently."""
    conn = connection or get_redis_connection()
    release_marker = f"{ADMISSION_RELEASED_PREFIX}{job_id}"
    try:
        first_release = bool(
            conn.set(
                release_marker,
                "1",
                nx=True,
                ex=_ADMISSION_RELEASED_TTL_SECONDS,
            )
        )
    except Exception as exc:
        logger.warning("Failed to mark admission release for %s: %s", job_id, exc)
        return False

    if not first_release:
        return False

    resolved_group = job_group
    if not resolved_group:
        resolved_group = _decode_redis_value(conn.get(f"{ADMISSION_JOB_GROUP_PREFIX}{job_id}"))

    if resolved_group in _ADMISSION_KEY_BY_GROUP:
        admission_key = _admission_key_for_group(resolved_group)
        script = """
local key = KEYS[1]
local current = tonumber(redis.call('GET', key) or '0')
if current > 0 then
  return redis.call('DECR', key)
end
return 0
"""
        try:
            conn.eval(script, 1, admission_key)
        except Exception as exc:
            logger.warning(
                "Failed to decrement admission counter for %s (%s): %s",
                job_id,
                resolved_group,
                exc,
            )
        _increment_admission_release_counters(conn=conn, group_key=resolved_group)

    try:
        conn.delete(f"{ADMISSION_JOB_GROUP_PREFIX}{job_id}")
    except Exception:
        pass
    return True


def get_admission_counts(conn: redis.Redis | None = None) -> dict[str, int]:
    c = conn or get_redis_connection()

    def _as_int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "video": _as_int(c.get(ADMISSION_VIDEO_KEY)),
        "clip": _as_int(c.get(ADMISSION_CLIP_KEY)),
        "social": _as_int(c.get(ADMISSION_SOCIAL_KEY)),
        "releases_total": _as_int(c.get(ADMISSION_RELEASES_TOTAL_KEY)),
        "releases_video": _as_int(c.get(ADMISSION_RELEASES_VIDEO_KEY)),
        "releases_clip": _as_int(c.get(ADMISSION_RELEASES_CLIP_KEY)),
        "releases_social": _as_int(c.get(ADMISSION_RELEASES_SOCIAL_KEY)),
    }


def reconcile_admission_counts(conn: redis.Redis | None = None) -> dict[str, int]:
    """Best-effort reconcile admission counters to current queue depth totals."""
    c = conn or get_redis_connection()
    reconciled: dict[str, int] = {}
    for group_key, (priority_q, normal_q, max_depth) in _QUEUE_GROUPS.items():
        total = int(c.llen(f"rq:queue:{priority_q}")) + int(c.llen(f"rq:queue:{normal_q}"))
        normalized = max(0, min(int(max_depth), total))
        c.set(_admission_key_for_group(group_key), normalized)
        reconciled[group_key] = normalized
    return reconciled


def enqueue_job(queue_name: str, task_path: str, job_data: dict, *, job_id: str):
    """Enqueue a job after atomic admission control."""
    conn = get_redis_connection()
    group_key = admission_group_for_queue(queue_name)
    admission_token: int | None = None
    if group_key is not None:
        _, _, max_depth = _QUEUE_GROUPS[group_key]
        admission_token = _acquire_admission_token(
            conn=conn,
            group_key=group_key,
            max_depth=max_depth,
        )
        conn.set(
            f"{ADMISSION_JOB_GROUP_PREFIX}{job_id}",
            group_key,
            ex=_ADMISSION_JOB_GROUP_TTL_SECONDS,
        )

    queue = get_queue(queue_name, conn)
    job_meta = {
        "admission_group": group_key,
        "admission_token": admission_token,
    }
    retry_policy = Retry(max=_DEFAULT_RETRY_MAX, interval=list(_DEFAULT_RETRY_INTERVALS_SECONDS))
    try:
        return queue.enqueue(
            task_path,
            job_data,
            job_id=job_id,
            meta=job_meta,
            retry=retry_policy,
        )
    except Exception:
        if group_key is not None:
            _rollback_admission(conn=conn, group_key=group_key, job_id=job_id)
        raise


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
    default_social: int = 0,
) -> tuple[int, int, int]:
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
    social_workers = get_group_worker_scale_target(
        group="social",
        connection=conn,
        default=default_social,
    )
    return video_workers, clip_workers, social_workers


def set_worker_scale_target(
    *,
    video_workers: int | None = None,
    clip_workers: int | None = None,
    social_workers: int | None = None,
    connection: redis.Redis | None = None,
    default_video: int = 0,
    default_clip: int = 0,
    default_social: int = 0,
) -> tuple[int, int, int]:
    """Persist desired worker counts and return the normalized final values."""
    if video_workers is None and clip_workers is None and social_workers is None:
        raise ValueError("At least one worker count must be provided")

    conn = connection or get_redis_connection()
    current_video, current_clip, current_social = get_worker_scale_target(
        connection=conn,
        default_video=default_video,
        default_clip=default_clip,
        default_social=default_social,
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
    final_social = (
        set_group_worker_scale_target(
            group="social",
            workers=social_workers,
            connection=conn,
            default=current_social,
        )
        if social_workers is not None
        else current_social
    )

    return final_video, final_clip, final_social


# ---------------------------------------------------------------------------
# Module-level singletons (used by api.py - single process)
# ---------------------------------------------------------------------------
redis_conn = get_redis_connection()
video_priority_queue = get_queue("video-processing-priority", redis_conn)
video_queue = get_queue("video-processing", redis_conn)
clip_priority_queue = get_queue("clip-generation-priority", redis_conn)
clip_queue = get_queue("clip-generation", redis_conn)
social_priority_queue = get_queue("social-publishing-priority", redis_conn)
social_queue = get_queue("social-publishing", redis_conn)

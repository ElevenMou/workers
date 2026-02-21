import os

import redis
from rq import Queue
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, VIDEO_JOB_TIMEOUT, CLIP_JOB_TIMEOUT

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

# ---------------------------------------------------------------------------
# Queue configuration - maps queue name → RQ Queue kwargs
# ---------------------------------------------------------------------------
QUEUE_CONFIG: dict[str, dict] = {
    "video-processing": {"default_timeout": VIDEO_JOB_TIMEOUT},
    "clip-generation": {"default_timeout": CLIP_JOB_TIMEOUT},
}

WORKER_SCALE_KEY = "clipry:workers:desired"
MAX_WORKER_COUNT = 64


# ---------------------------------------------------------------------------
# Factory helpers (safe across subprocesses - each gets its own connection)
# ---------------------------------------------------------------------------
def get_redis_connection() -> redis.Redis:
    """Create a **new** Redis connection."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        socket_connect_timeout=5,
        # Keep blocking worker operations (BLPOP/PUBSUB) alive.
        socket_timeout=None,
        retry_on_timeout=True,
    )


def get_queue(name: str, connection: redis.Redis | None = None) -> Queue:
    """Create a single :class:`Queue` with its configured timeout."""
    conn = connection or get_redis_connection()
    cfg = QUEUE_CONFIG.get(name, {})
    return Queue(name, connection=conn, default_timeout=cfg.get("default_timeout", 600))


def get_queues(names: list[str], connection: redis.Redis | None = None) -> list[Queue]:
    """Create multiple queues sharing one Redis connection."""
    conn = connection or get_redis_connection()
    return [get_queue(n, conn) for n in names]


def enqueue_job(queue_name: str, task_path: str, job_data: dict, *, job_id: str):
    """Enqueue using a fresh Redis connection to avoid stale pooled sockets."""
    queue = get_queue(queue_name)
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


def get_worker_scale_target(
    *,
    connection: redis.Redis | None = None,
    default_video: int = 0,
    default_clip: int = 0,
) -> tuple[int, int]:
    """Read desired worker counts from Redis, falling back to defaults."""
    conn = connection or get_redis_connection()
    data = conn.hgetall(WORKER_SCALE_KEY) or {}

    video_raw = data.get(b"video_workers") or data.get("video_workers")
    clip_raw = data.get(b"clip_workers") or data.get("clip_workers")

    video_workers = _normalize_worker_count(
        _decode_redis_value(video_raw),
        default_video,
    )
    clip_workers = _normalize_worker_count(
        _decode_redis_value(clip_raw),
        default_clip,
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
        _normalize_worker_count(video_workers, current_video)
        if video_workers is not None
        else current_video
    )
    final_clip = (
        _normalize_worker_count(clip_workers, current_clip)
        if clip_workers is not None
        else current_clip
    )

    conn.hset(
        WORKER_SCALE_KEY,
        mapping={
            "video_workers": final_video,
            "clip_workers": final_clip,
        },
    )
    return final_video, final_clip


# ---------------------------------------------------------------------------
# Module-level singletons (used by api.py - single process)
# ---------------------------------------------------------------------------
redis_conn = get_redis_connection()
video_queue = get_queue("video-processing", redis_conn)
clip_queue = get_queue("clip-generation", redis_conn)

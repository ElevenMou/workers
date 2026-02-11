import redis
from rq import Queue
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, VIDEO_JOB_TIMEOUT, CLIP_JOB_TIMEOUT

# ---------------------------------------------------------------------------
# Queue configuration - maps queue name → RQ Queue kwargs
# ---------------------------------------------------------------------------
QUEUE_CONFIG: dict[str, dict] = {
    "video-processing": {"default_timeout": VIDEO_JOB_TIMEOUT},
    "clip-generation": {"default_timeout": CLIP_JOB_TIMEOUT},
}


# ---------------------------------------------------------------------------
# Factory helpers (safe across subprocesses - each gets its own connection)
# ---------------------------------------------------------------------------
def get_redis_connection() -> redis.Redis:
    """Create a **new** Redis connection."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        socket_connect_timeout=5,
        socket_timeout=30,
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


# ---------------------------------------------------------------------------
# Module-level singletons (used by api.py - single process)
# ---------------------------------------------------------------------------
redis_conn = get_redis_connection()
video_queue = get_queue("video-processing", redis_conn)
clip_queue = get_queue("clip-generation", redis_conn)

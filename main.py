"""
Clipry worker pool.

Spawns separate subprocesses for *video-processing* and *clip-generation*
queues so multiple jobs run concurrently.  The number of workers per queue
is controlled by ``NUM_VIDEO_WORKERS`` / ``NUM_CLIP_WORKERS`` env vars
(see config.py).

    python main.py          # launch the full pool
"""

import logging
import multiprocessing
import os
import sys

from rq import Worker, SimpleWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s [%(process)d] %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.worker")


# ---------------------------------------------------------------------------
# Windows compatibility – SimpleWorker without SIGALRM
# ---------------------------------------------------------------------------
class _NoOpDeathPenalty:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _WindowsWorker(SimpleWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.death_penalty_class = _NoOpDeathPenalty


def _worker_cls():
    return _WindowsWorker if os.name == "nt" else Worker


# ---------------------------------------------------------------------------
# Subprocess entry-point
# ---------------------------------------------------------------------------
def _run_worker(queue_names: list[str], worker_name: str):
    """Start a single RQ worker inside a subprocess.

    Each subprocess creates its **own** Redis connection so there is no
    shared state between processes.
    """
    # Re-import inside the subprocess (required for Windows 'spawn')
    from utils.redis_client import get_redis_connection, get_queues  # noqa: E402

    conn = get_redis_connection()
    queues = get_queues(queue_names, conn)
    cls = _worker_cls()

    logger.info("Worker %s starting – queues: %s", worker_name, queue_names)
    worker = cls(queues, connection=conn, name=worker_name)
    worker.work(with_scheduler=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Import config here (not at module level) to avoid heavy transitive
    # imports in every subprocess before they actually need them.
    from config import NUM_VIDEO_WORKERS, NUM_CLIP_WORKERS

    processes: list[multiprocessing.Process] = []

    # -- Video-processing workers -----------------------------------------
    for i in range(NUM_VIDEO_WORKERS):
        p = multiprocessing.Process(
            target=_run_worker,
            args=(["video-processing"], f"video-worker-{i}"),
            daemon=True,
        )
        p.start()
        processes.append(p)
        logger.info("Spawned video-worker-%d  (pid %d)", i, p.pid)

    # -- Clip-generation workers ------------------------------------------
    for i in range(NUM_CLIP_WORKERS):
        p = multiprocessing.Process(
            target=_run_worker,
            args=(["clip-generation"], f"clip-worker-{i}"),
            daemon=True,
        )
        p.start()
        processes.append(p)
        logger.info("Spawned clip-worker-%d  (pid %d)", i, p.pid)

    total = NUM_VIDEO_WORKERS + NUM_CLIP_WORKERS
    logger.info(
        "Worker pool ready – %d video, %d clip  (%d total)",
        NUM_VIDEO_WORKERS,
        NUM_CLIP_WORKERS,
        total,
    )

    # -- Keep the supervisor alive; Ctrl-C for graceful shutdown ----------
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutting down workers …")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=10)
        logger.info("All workers stopped.")
        sys.exit(0)

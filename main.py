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
import time

from rq import Worker, SimpleWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s [%(process)d] %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.worker")

# ---------------------------------------------------------------------------
# Windows compatibility - SimpleWorker without SIGALRM
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
# Stale-worker cleanup
# ---------------------------------------------------------------------------
def _cleanup_named_workers(conn, worker_names: set[str]):
    """Remove Redis worker registrations for the exact provided names.

    RQ stores worker metadata in Redis.  If the supervisor was killed without
    a clean shutdown the old registrations persist and new workers with the
    same names are rejected with ``ValueError``.
    """
    try:
        existing = Worker.all(connection=conn)
    except Exception as exc:
        logger.warning("Could not list existing workers: %s", exc)
        return

    for w in existing:
        if w.name not in worker_names:
            continue

        logger.info("Deregistering stale worker: %s (pid=%s)", w.name, w.pid)
        try:
            pipe = conn.pipeline()
            pipe.delete(w.key)
            pipe.srem("rq:workers", w.key)
            for q in w.queues:
                pipe.srem(f"rq:workers:{q.name}", w.key)
            pipe.execute()
        except Exception as exc:
            logger.warning("Failed to deregister %s: %s", w.name, exc)


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

    logger.info("Worker %s starting - queues: %s", worker_name, queue_names)
    worker = cls(queues, connection=conn, name=worker_name)
    worker.work(with_scheduler=False)


def _spawn_worker(queue_names: list[str], worker_name: str) -> multiprocessing.Process:
    p = multiprocessing.Process(
        target=_run_worker,
        args=(queue_names, worker_name),
        daemon=False,
    )
    p.start()
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Import config here (not at module level) to avoid heavy transitive
    # imports in every subprocess before they actually need them.
    from config import NUM_VIDEO_WORKERS, NUM_CLIP_WORKERS, validate_env
    from utils.redis_client import get_redis_connection

    validate_env()

    worker_specs: dict[str, list[str]] = {}
    processes: dict[str, multiprocessing.Process] = {}

    # Build the full worker name set first so cleanup can target exact names.
    for i in range(NUM_VIDEO_WORKERS):
        worker_specs[f"video-worker-{i}"] = ["video-processing"]
    for i in range(NUM_CLIP_WORKERS):
        worker_specs[f"clip-worker-{i}"] = ["clip-generation"]

    # -- Clean up stale workers with the same names -----------------------
    _startup_conn = get_redis_connection()
    _cleanup_named_workers(_startup_conn, set(worker_specs.keys()))

    # -- Video-processing workers -----------------------------------------
    for i in range(NUM_VIDEO_WORKERS):
        name = f"video-worker-{i}"
        queues = worker_specs[name]
        p = _spawn_worker(queues, name)
        processes[name] = p
        logger.info("Spawned video-worker-%d  (pid %d)", i, p.pid)

    # -- Clip-generation workers ------------------------------------------
    for i in range(NUM_CLIP_WORKERS):
        name = f"clip-worker-{i}"
        queues = worker_specs[name]
        p = _spawn_worker(queues, name)
        processes[name] = p
        logger.info("Spawned clip-worker-%d  (pid %d)", i, p.pid)

    total = NUM_VIDEO_WORKERS + NUM_CLIP_WORKERS
    logger.info(
        "Worker pool ready - %d video, %d clip  (%d total)",
        NUM_VIDEO_WORKERS,
        NUM_CLIP_WORKERS,
        total,
    )

    # -- Keep the supervisor alive; Ctrl-C for graceful shutdown ----------
    try:
        while True:
            for name, p in list(processes.items()):
                if p.is_alive():
                    continue

                exit_code = p.exitcode
                logger.warning(
                    "Worker %s exited unexpectedly (exit code %s). Restarting.",
                    name,
                    exit_code,
                )
                _cleanup_named_workers(_startup_conn, {name})
                replacement = _spawn_worker(worker_specs[name], name)
                processes[name] = replacement
                logger.info("Respawned %s (pid %d)", name, replacement.pid)

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down workers …")

        # Terminate subprocesses
        for p in processes.values():
            p.terminate()
        for p in processes.values():
            p.join(timeout=10)

        # Deregister from Redis so the next startup is clean
        _cleanup_named_workers(_startup_conn, set(worker_specs.keys()))
        logger.info("All workers stopped.")
        sys.exit(0)

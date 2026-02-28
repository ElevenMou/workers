"""Worker process lifecycle operations for RQ supervisor."""

import multiprocessing
import os
import threading

from rq import SimpleWorker, Worker

from worker_supervisor.heartbeat import beat, clear_heartbeat
from worker_supervisor.state import logger


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


def worker_cls():
    """Select worker implementation by platform."""
    return _WindowsWorker if os.name == "nt" else Worker


def _worker_key(worker_name: str) -> str:
    return f"{Worker.redis_worker_namespace_prefix}{worker_name}"


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _force_remove_worker_registration(conn, worker_name: str):
    """Delete worker hash + membership sets by name, even if Worker.all misses it."""
    key = _worker_key(worker_name)
    queue_keys = []
    try:
        queue_keys = list(conn.smembers("rq:queues") or [])
    except Exception as exc:
        logger.warning("Could not enumerate queues while cleaning %s: %s", worker_name, exc)

    try:
        pipe = conn.pipeline()
        pipe.delete(key)
        pipe.srem("rq:workers", key)
        for queue_key in queue_keys:
            queue_value = _decode(queue_key)
            if not queue_value.startswith("rq:queue:"):
                continue
            queue_name = queue_value[len("rq:queue:") :]
            if queue_name:
                pipe.srem(f"rq:workers:{queue_name}", key)
        pipe.execute()
    except Exception as exc:
        logger.warning("Failed direct worker cleanup for %s: %s", worker_name, exc)


def cleanup_named_workers(conn, worker_names: set[str]):
    """Remove Redis worker registrations for exact provided names."""
    for name in worker_names:
        _force_remove_worker_registration(conn, name)

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


_HEARTBEAT_INTERVAL_SECONDS = 30


def _heartbeat_loop(conn, worker_name: str, stop_event: threading.Event) -> None:
    """Background thread that sends periodic heartbeats to Redis."""
    while not stop_event.wait(timeout=_HEARTBEAT_INTERVAL_SECONDS):
        try:
            beat(conn, worker_name)
        except Exception:
            pass  # Best-effort; supervisor will detect missing beats


def run_worker(queue_names: list[str], worker_name: str):
    """Start a single RQ worker inside a subprocess."""
    from utils.redis_client import get_queues, get_redis_connection  # noqa: E402

    conn = get_redis_connection()
    queues = get_queues(queue_names, conn)
    cls = worker_cls()

    should_preload_whisper = (
        os.getenv("PRELOAD_WHISPER_ON_WORKER_START", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    has_video_work = any(str(name).startswith("video-processing") for name in queue_names)
    if should_preload_whisper and has_video_work:
        try:
            from services.transcriber import Transcriber  # noqa: E402

            Transcriber()
            logger.info("Worker %s preloaded Whisper model", worker_name)
        except Exception as exc:
            logger.warning("Worker %s failed to preload Whisper model: %s", worker_name, exc)

    # Start heartbeat thread so the supervisor can detect hung workers
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(conn, worker_name, stop_event), daemon=True
    )
    hb_thread.start()
    beat(conn, worker_name)  # Immediate first heartbeat

    logger.info("Worker %s starting - queues: %s", worker_name, queue_names)
    try:
        for attempt in range(2):
            worker = cls(queues, connection=conn, name=worker_name)
            try:
                worker.work(with_scheduler=False)
                return
            except ValueError as exc:
                if "There exists an active worker named" not in str(exc) or attempt > 0:
                    raise
                logger.warning(
                    "Detected worker-name collision for %s; forcing Redis cleanup and retrying once.",
                    worker_name,
                )
                _force_remove_worker_registration(conn, worker_name)
    finally:
        stop_event.set()
        hb_thread.join(timeout=5)
        try:
            clear_heartbeat(conn, worker_name)
        except Exception:
            pass


def spawn_worker(queue_names: list[str], worker_name: str) -> multiprocessing.Process:
    """Create and start a worker subprocess."""
    p = multiprocessing.Process(
        target=run_worker,
        args=(queue_names, worker_name),
        daemon=False,
    )
    p.start()
    return p


def request_graceful_shutdown(conn, worker_name: str) -> bool:
    """Best-effort graceful stop via RQ control command."""
    try:
        from rq.command import send_shutdown_command

        send_shutdown_command(conn, worker_name)
        return True
    except Exception as exc:
        logger.warning("Graceful shutdown command failed for %s: %s", worker_name, exc)
        return False


def stop_worker_process(conn, name: str, process: multiprocessing.Process | None):
    """Stop worker process gracefully when possible; force-kill as fallback."""
    if process is None:
        cleanup_named_workers(conn, {name})
        return

    if process.is_alive():
        logger.info("Stopping %s (pid %s)", name, process.pid)

        graceful_sent = request_graceful_shutdown(conn, name)
        if graceful_sent:
            process.join(timeout=15)

        if process.is_alive():
            logger.warning("Force terminating %s (pid %s)", name, process.pid)
            process.terminate()
            process.join(timeout=10)
    else:
        process.join(timeout=1)

    cleanup_named_workers(conn, {name})


def resize_group(
    *,
    conn,
    worker_specs: dict[str, list[str]],
    processes: dict[str, multiprocessing.Process],
    prefix: str,
    queue_names: list[str],
    current_count: int,
    target_count: int,
) -> int:
    """Scale a worker group up/down to target_count."""
    target_count = max(0, int(target_count))
    if target_count == current_count:
        return current_count

    if target_count > current_count:
        for i in range(current_count, target_count):
            name = f"{prefix}-{i}"
            worker_specs[name] = queue_names
            cleanup_named_workers(conn, {name})
            p = spawn_worker(queue_names, name)
            processes[name] = p
            logger.info("Scaled up: spawned %s (pid %d)", name, p.pid)
    else:
        for i in range(current_count - 1, target_count - 1, -1):
            name = f"{prefix}-{i}"
            p = processes.pop(name, None)
            worker_specs.pop(name, None)
            stop_worker_process(conn, name, p)

    return target_count

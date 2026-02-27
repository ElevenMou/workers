"""Supervisor loop that manages all worker subprocesses."""

import multiprocessing
import os
import re
import socket
import sys
import time
import uuid

from worker_supervisor.runtime import (
    cleanup_named_workers,
    resize_group,
    spawn_worker,
    stop_worker_process,
)
from worker_supervisor.startup import env_bool, recover_stale_processing_rows
from worker_supervisor.state import logger

VIDEO_PRIORITY_QUEUE = "video-processing-priority"
VIDEO_QUEUE = "video-processing"
CLIP_PRIORITY_QUEUE = "clip-generation-priority"
CLIP_QUEUE = "clip-generation"
VIDEO_QUEUE_ORDER = [VIDEO_PRIORITY_QUEUE, VIDEO_QUEUE]
CLIP_QUEUE_ORDER = [CLIP_PRIORITY_QUEUE, CLIP_QUEUE]
VIDEO_PREFIX = "video-worker"
CLIP_PREFIX = "clip-worker"
MAINTENANCE_LEADER_KEY = "clipry:maintenance:leader"


def _sanitize_instance_id(raw: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw or "").strip())
    token = token.strip("-_")
    if not token:
        token = "instance"
    return token[:64]


def _resolve_instance_id(configured_instance_id: str | None) -> str:
    if configured_instance_id:
        return _sanitize_instance_id(configured_instance_id)
    return _sanitize_instance_id(socket.gethostname() or f"node-{uuid.uuid4().hex[:8]}")


def _leader_lock_eval(conn, script: str, keys: list[str], args: list[object]) -> int:
    try:
        return int(conn.eval(script, len(keys), *keys, *args))
    except Exception:
        return 0


class _MaintenanceLeaderLock:
    def __init__(self, conn, *, ttl_seconds: int):
        self.conn = conn
        self.ttl_seconds = max(5, int(ttl_seconds))
        self.token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self.acquired = False

    def acquire(self) -> bool:
        try:
            self.acquired = bool(
                self.conn.set(
                    MAINTENANCE_LEADER_KEY,
                    self.token,
                    nx=True,
                    ex=self.ttl_seconds,
                )
            )
        except Exception as exc:
            logger.warning("Maintenance leader lock acquire failed: %s", exc)
            self.acquired = False
        return self.acquired

    def renew(self) -> bool:
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('EXPIRE', KEYS[1], ARGV[2]) "
            "else return 0 end"
        )
        renewed = _leader_lock_eval(
            self.conn,
            script,
            [MAINTENANCE_LEADER_KEY],
            [self.token, self.ttl_seconds],
        )
        self.acquired = renewed == 1
        return self.acquired

    def release(self) -> None:
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) "
            "else return 0 end"
        )
        _leader_lock_eval(
            self.conn,
            script,
            [MAINTENANCE_LEADER_KEY],
            [self.token],
        )
        self.acquired = False


def _run_maintenance_loop(
    *,
    startup_conn,
    queue_names: list[str],
    raw_video_cleanup_interval_seconds: int,
    clip_asset_cleanup_interval_seconds: int,
    processing_job_stale_seconds: int,
    leader_lock_ttl_seconds: int,
    leader_renew_seconds: int,
) -> int:
    from tasks.clips.cleanup import cleanup_expired_clip_assets
    from tasks.videos.cleanup import cleanup_expired_raw_videos

    lock = _MaintenanceLeaderLock(
        startup_conn,
        ttl_seconds=leader_lock_ttl_seconds,
    )

    last_raw_video_cleanup_run = 0.0
    last_clip_asset_cleanup_run = 0.0
    processing_recovery_interval_seconds = max(
        30,
        min(
            300,
            int(processing_job_stale_seconds // 4) if processing_job_stale_seconds else 60,
        ),
    )
    last_processing_recovery_run = 0.0
    last_lock_renewal = 0.0
    renew_every = max(1, int(leader_renew_seconds))
    has_logged_standby = False

    logger.info(
        "Supervisor role=maintenance started (leader_key=%s ttl=%ss renew=%ss)",
        MAINTENANCE_LEADER_KEY,
        max(5, int(leader_lock_ttl_seconds)),
        renew_every,
    )
    should_warm_whisper = (
        os.getenv("MAINTENANCE_WARM_WHISPER", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if should_warm_whisper:
        try:
            from services.transcriber import Transcriber

            Transcriber()
            logger.info("Maintenance warm-up check: Whisper model loaded")
        except Exception as exc:
            logger.warning("Maintenance warm-up check failed for Whisper model: %s", exc)

    try:
        while True:
            now = time.monotonic()
            if not lock.acquired:
                if lock.acquire():
                    has_logged_standby = False
                    last_lock_renewal = now
                    logger.info("Acquired maintenance leader lock")
                else:
                    if not has_logged_standby:
                        logger.info("Maintenance supervisor in standby (leader lock not acquired)")
                        has_logged_standby = True
                    time.sleep(1)
                    continue

            if now - last_lock_renewal >= renew_every:
                if not lock.renew():
                    logger.warning("Lost maintenance leader lock; returning to standby")
                    continue
                last_lock_renewal = now

            if (
                raw_video_cleanup_interval_seconds > 0
                and now - last_raw_video_cleanup_run >= raw_video_cleanup_interval_seconds
            ):
                try:
                    cleanup_expired_raw_videos()
                except Exception as exc:
                    logger.warning("Expired raw-video cleanup tick failed: %s", exc)
                finally:
                    last_raw_video_cleanup_run = now

            if (
                clip_asset_cleanup_interval_seconds > 0
                and now - last_clip_asset_cleanup_run >= clip_asset_cleanup_interval_seconds
            ):
                try:
                    cleanup_expired_clip_assets()
                except Exception as exc:
                    logger.warning("Expired clip-asset cleanup tick failed: %s", exc)
                finally:
                    last_clip_asset_cleanup_run = now

            if (
                processing_job_stale_seconds > 0
                and now - last_processing_recovery_run >= processing_recovery_interval_seconds
            ):
                try:
                    recover_stale_processing_rows(
                        conn=startup_conn,
                        queue_names=queue_names,
                        stale_seconds=processing_job_stale_seconds,
                    )
                except Exception as exc:
                    logger.warning("Runtime stale-processing recovery tick failed: %s", exc)
                finally:
                    last_processing_recovery_run = now

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down maintenance supervisor ...")
        lock.release()
        return 0
    except Exception:
        lock.release()
        logger.exception("Maintenance supervisor crashed unexpectedly")
        return 1


def run_supervisor() -> int:
    """Run the worker supervisor until interrupted."""
    from config import (
        CLIP_ASSET_CLEANUP_INTERVAL_SECONDS,
        MAINTENANCE_LEADER_LOCK_TTL_SECONDS,
        MAINTENANCE_LEADER_RENEW_SECONDS,
        NUM_CLIP_WORKERS,
        NUM_VIDEO_WORKERS,
        PROCESSING_JOB_STALE_SECONDS,
        RAW_VIDEO_CLEANUP_INTERVAL_SECONDS,
        SUPERVISOR_ROLE,
        WORKER_INSTANCE_ID,
        validate_env,
    )
    from utils.redis_client import (
        get_redis_connection,
        get_group_worker_scale_target,
        reconcile_admission_counts,
        get_worker_scale_target,
        set_group_worker_scale_target,
        set_worker_scale_target,
    )

    validate_env()
    supervisor_role = str(SUPERVISOR_ROLE or "worker").strip().lower() or "worker"
    if supervisor_role not in {"worker", "maintenance"}:
        logger.error(
            "Invalid SUPERVISOR_ROLE=%r. Expected 'worker' or 'maintenance'.",
            supervisor_role,
        )
        return 1

    instance_id = _resolve_instance_id(WORKER_INSTANCE_ID)
    queue_names = [
        VIDEO_PRIORITY_QUEUE,
        VIDEO_QUEUE,
        CLIP_PRIORITY_QUEUE,
        CLIP_QUEUE,
    ]
    startup_conn = get_redis_connection()

    if supervisor_role == "maintenance":
        return _run_maintenance_loop(
            startup_conn=startup_conn,
            queue_names=queue_names,
            raw_video_cleanup_interval_seconds=RAW_VIDEO_CLEANUP_INTERVAL_SECONDS,
            clip_asset_cleanup_interval_seconds=CLIP_ASSET_CLEANUP_INTERVAL_SECONDS,
            processing_job_stale_seconds=PROCESSING_JOB_STALE_SECONDS,
            leader_lock_ttl_seconds=MAINTENANCE_LEADER_LOCK_TTL_SECONDS,
            leader_renew_seconds=MAINTENANCE_LEADER_RENEW_SECONDS,
        )

    try:
        reconciled = reconcile_admission_counts(startup_conn)
        logger.info(
            "Reconciled admission counters at startup: video=%d clip=%d",
            int(reconciled.get("video", 0)),
            int(reconciled.get("clip", 0)),
        )
    except Exception as exc:
        logger.warning("Admission-counter reconciliation failed at startup: %s", exc)

    # Optional WORKER_MODE restricts this supervisor to one worker group.
    worker_mode = (os.getenv("WORKER_MODE") or "").strip().lower()
    if worker_mode == "video":
        effective_num_video = NUM_VIDEO_WORKERS
        effective_num_clip = 0
        logger.info("WORKER_MODE=video - only video workers will be managed")
    elif worker_mode == "clip":
        effective_num_video = 0
        effective_num_clip = NUM_CLIP_WORKERS
        logger.info("WORKER_MODE=clip - only clip workers will be managed")
    else:
        effective_num_video = NUM_VIDEO_WORKERS
        effective_num_clip = NUM_CLIP_WORKERS

    worker_specs: dict[str, list[str]] = {}
    processes: dict[str, multiprocessing.Process] = {}

    if env_bool("RESET_WORKER_SCALE_ON_START", True):
        if worker_mode == "video":
            desired_video_workers = set_group_worker_scale_target(
                group="video",
                workers=effective_num_video,
                connection=startup_conn,
                default=effective_num_video,
            )
            desired_clip_workers = 0
        elif worker_mode == "clip":
            desired_video_workers = 0
            desired_clip_workers = set_group_worker_scale_target(
                group="clip",
                workers=effective_num_clip,
                connection=startup_conn,
                default=effective_num_clip,
            )
        else:
            desired_video_workers, desired_clip_workers = set_worker_scale_target(
                connection=startup_conn,
                video_workers=effective_num_video,
                clip_workers=effective_num_clip,
                default_video=effective_num_video,
                default_clip=effective_num_clip,
            )
        logger.info(
            "Reset worker scale target on startup to defaults: video=%d clip=%d",
            desired_video_workers,
            desired_clip_workers,
        )
    else:
        if worker_mode == "video":
            desired_video_workers = get_group_worker_scale_target(
                group="video",
                connection=startup_conn,
                default=effective_num_video,
            )
            desired_clip_workers = 0
        elif worker_mode == "clip":
            desired_video_workers = 0
            desired_clip_workers = get_group_worker_scale_target(
                group="clip",
                connection=startup_conn,
                default=effective_num_clip,
            )
        else:
            desired_video_workers, desired_clip_workers = get_worker_scale_target(
                connection=startup_conn,
                default_video=effective_num_video,
                default_clip=effective_num_clip,
            )

    # Enforce WORKER_MODE limits even when reading from Redis.
    if worker_mode == "video":
        desired_clip_workers = 0
    elif worker_mode == "clip":
        desired_video_workers = 0

    for i in range(desired_video_workers):
        worker_specs[f"{VIDEO_PREFIX}-{instance_id}-{i}"] = list(VIDEO_QUEUE_ORDER)
    for i in range(desired_clip_workers):
        worker_specs[f"{CLIP_PREFIX}-{instance_id}-{i}"] = list(CLIP_QUEUE_ORDER)

    cleanup_named_workers(startup_conn, set(worker_specs.keys()))
    logger.info(
        "Distributed-safe worker mode active: startup DB/queue recovery sweeps are disabled "
        "(instance_id=%s role=%s).",
        instance_id,
        supervisor_role,
    )

    for i in range(desired_video_workers):
        name = f"{VIDEO_PREFIX}-{instance_id}-{i}"
        p = spawn_worker(worker_specs[name], name)
        processes[name] = p
        logger.info("Spawned %s  (pid %d)", name, p.pid)

    for i in range(desired_clip_workers):
        name = f"{CLIP_PREFIX}-{instance_id}-{i}"
        p = spawn_worker(worker_specs[name], name)
        processes[name] = p
        logger.info("Spawned %s  (pid %d)", name, p.pid)

    current_video_workers = desired_video_workers
    current_clip_workers = desired_clip_workers

    total = current_video_workers + current_clip_workers
    logger.info(
        "Worker pool ready - instance=%s %d video, %d clip  (%d total)",
        instance_id,
        current_video_workers,
        current_clip_workers,
        total,
    )

    try:
        while True:
            try:
                if worker_mode == "video":
                    target_video_workers = get_group_worker_scale_target(
                        group="video",
                        connection=startup_conn,
                        default=current_video_workers,
                    )
                    target_clip_workers = 0
                elif worker_mode == "clip":
                    target_video_workers = 0
                    target_clip_workers = get_group_worker_scale_target(
                        group="clip",
                        connection=startup_conn,
                        default=current_clip_workers,
                    )
                else:
                    target_video_workers, target_clip_workers = get_worker_scale_target(
                        connection=startup_conn,
                        default_video=current_video_workers,
                        default_clip=current_clip_workers,
                    )
            except Exception as exc:
                logger.warning("Failed to read scale target from Redis: %s", exc)
                target_video_workers = current_video_workers
                target_clip_workers = current_clip_workers

            if (
                target_video_workers != current_video_workers
                or target_clip_workers != current_clip_workers
            ):
                logger.info(
                    "Applying scale target - video: %d -> %d, clip: %d -> %d",
                    current_video_workers,
                    target_video_workers,
                    current_clip_workers,
                    target_clip_workers,
                )
                current_video_workers = resize_group(
                    conn=startup_conn,
                    worker_specs=worker_specs,
                    processes=processes,
                    prefix=f"{VIDEO_PREFIX}-{instance_id}",
                    queue_names=list(VIDEO_QUEUE_ORDER),
                    current_count=current_video_workers,
                    target_count=target_video_workers,
                )
                current_clip_workers = resize_group(
                    conn=startup_conn,
                    worker_specs=worker_specs,
                    processes=processes,
                    prefix=f"{CLIP_PREFIX}-{instance_id}",
                    queue_names=list(CLIP_QUEUE_ORDER),
                    current_count=current_clip_workers,
                    target_count=target_clip_workers,
                )

            for name, p in list(processes.items()):
                if p.is_alive():
                    continue

                exit_code = p.exitcode
                logger.warning(
                    "Worker %s exited unexpectedly (exit code %s). Restarting.",
                    name,
                    exit_code,
                )
                cleanup_named_workers(startup_conn, {name})
                replacement = spawn_worker(worker_specs[name], name)
                processes[name] = replacement
                logger.info("Respawned %s (pid %d)", name, replacement.pid)

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down workers ...")

        for name, p in list(processes.items()):
            stop_worker_process(startup_conn, name, p)

        cleanup_named_workers(startup_conn, set(worker_specs.keys()))
        logger.info("All workers stopped.")
        return 0
    except Exception:
        logger.exception("Supervisor crashed unexpectedly")
        return 1


def main() -> int:
    """CLI entrypoint helper."""
    return run_supervisor()


if __name__ == "__main__":
    sys.exit(main())

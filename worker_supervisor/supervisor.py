"""Supervisor loop that manages all worker subprocesses."""

import multiprocessing
import sys
import time

from worker_supervisor.runtime import (
    cleanup_named_workers,
    resize_group,
    spawn_worker,
    stop_worker_process,
)
from worker_supervisor.startup import (
    cleanup_started_jobs_on_start,
    env_bool,
    purge_startup_backlog,
    recover_processing_rows_on_start,
    recover_stale_processing_rows,
)
from worker_supervisor.state import logger

VIDEO_PRIORITY_QUEUE = "video-processing-priority"
VIDEO_QUEUE = "video-processing"
CLIP_PRIORITY_QUEUE = "clip-generation-priority"
CLIP_QUEUE = "clip-generation"
VIDEO_QUEUE_ORDER = [VIDEO_PRIORITY_QUEUE, VIDEO_QUEUE]
CLIP_QUEUE_ORDER = [CLIP_PRIORITY_QUEUE, CLIP_QUEUE]
VIDEO_PREFIX = "video-worker"
CLIP_PREFIX = "clip-worker"


def run_supervisor() -> int:
    """Run the worker supervisor until interrupted."""
    from config import (
        CLIP_ASSET_CLEANUP_INTERVAL_SECONDS,
        NUM_CLIP_WORKERS,
        NUM_VIDEO_WORKERS,
        PROCESSING_JOB_STALE_SECONDS,
        RAW_VIDEO_CLEANUP_INTERVAL_SECONDS,
        validate_env,
    )
    from tasks.clips.cleanup import cleanup_expired_clip_assets
    from tasks.videos.cleanup import cleanup_expired_raw_videos
    from utils.redis_client import (
        get_redis_connection,
        get_worker_scale_target,
        set_worker_scale_target,
    )

    validate_env()

    worker_specs: dict[str, list[str]] = {}
    processes: dict[str, multiprocessing.Process] = {}
    startup_conn = get_redis_connection()

    if env_bool("RESET_WORKER_SCALE_ON_START", True):
        desired_video_workers, desired_clip_workers = set_worker_scale_target(
            connection=startup_conn,
            video_workers=NUM_VIDEO_WORKERS,
            clip_workers=NUM_CLIP_WORKERS,
            default_video=NUM_VIDEO_WORKERS,
            default_clip=NUM_CLIP_WORKERS,
        )
        logger.info(
            "Reset worker scale target on startup to defaults: video=%d clip=%d",
            desired_video_workers,
            desired_clip_workers,
        )
    else:
        desired_video_workers, desired_clip_workers = get_worker_scale_target(
            connection=startup_conn,
            default_video=NUM_VIDEO_WORKERS,
            default_clip=NUM_CLIP_WORKERS,
        )

    for i in range(desired_video_workers):
        worker_specs[f"{VIDEO_PREFIX}-{i}"] = list(VIDEO_QUEUE_ORDER)
    for i in range(desired_clip_workers):
        worker_specs[f"{CLIP_PREFIX}-{i}"] = list(CLIP_QUEUE_ORDER)

    queue_names = [
        VIDEO_PRIORITY_QUEUE,
        VIDEO_QUEUE,
        CLIP_PRIORITY_QUEUE,
        CLIP_QUEUE,
    ]
    cleanup_named_workers(startup_conn, set(worker_specs.keys()))
    cleanup_started_jobs_on_start(startup_conn, queue_names)
    recover_processing_rows_on_start()
    purge_startup_backlog(startup_conn, queue_names)

    for i in range(desired_video_workers):
        name = f"{VIDEO_PREFIX}-{i}"
        p = spawn_worker(worker_specs[name], name)
        processes[name] = p
        logger.info("Spawned %s  (pid %d)", name, p.pid)

    for i in range(desired_clip_workers):
        name = f"{CLIP_PREFIX}-{i}"
        p = spawn_worker(worker_specs[name], name)
        processes[name] = p
        logger.info("Spawned %s  (pid %d)", name, p.pid)

    current_video_workers = desired_video_workers
    current_clip_workers = desired_clip_workers

    total = current_video_workers + current_clip_workers
    logger.info(
        "Worker pool ready - %d video, %d clip  (%d total)",
        current_video_workers,
        current_clip_workers,
        total,
    )
    last_raw_video_cleanup_run = 0.0
    last_clip_asset_cleanup_run = 0.0
    processing_recovery_interval_seconds = max(
        30,
        min(300, int(PROCESSING_JOB_STALE_SECONDS // 4) if PROCESSING_JOB_STALE_SECONDS else 60),
    )
    last_processing_recovery_run = 0.0

    try:
        while True:
            now = time.monotonic()
            if (
                RAW_VIDEO_CLEANUP_INTERVAL_SECONDS > 0
                and now - last_raw_video_cleanup_run >= RAW_VIDEO_CLEANUP_INTERVAL_SECONDS
            ):
                try:
                    cleanup_expired_raw_videos()
                except Exception as exc:
                    logger.warning("Expired raw-video cleanup tick failed: %s", exc)
                finally:
                    last_raw_video_cleanup_run = now

            if (
                CLIP_ASSET_CLEANUP_INTERVAL_SECONDS > 0
                and now - last_clip_asset_cleanup_run >= CLIP_ASSET_CLEANUP_INTERVAL_SECONDS
            ):
                try:
                    cleanup_expired_clip_assets()
                except Exception as exc:
                    logger.warning("Expired clip-asset cleanup tick failed: %s", exc)
                finally:
                    last_clip_asset_cleanup_run = now

            if (
                PROCESSING_JOB_STALE_SECONDS > 0
                and now - last_processing_recovery_run
                >= processing_recovery_interval_seconds
            ):
                try:
                    recover_stale_processing_rows(
                        conn=startup_conn,
                        queue_names=queue_names,
                        stale_seconds=PROCESSING_JOB_STALE_SECONDS,
                    )
                except Exception as exc:
                    logger.warning("Runtime stale-processing recovery tick failed: %s", exc)
                finally:
                    last_processing_recovery_run = now

            try:
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
                    prefix=VIDEO_PREFIX,
                    queue_names=list(VIDEO_QUEUE_ORDER),
                    current_count=current_video_workers,
                    target_count=target_video_workers,
                )
                current_clip_workers = resize_group(
                    conn=startup_conn,
                    worker_specs=worker_specs,
                    processes=processes,
                    prefix=CLIP_PREFIX,
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

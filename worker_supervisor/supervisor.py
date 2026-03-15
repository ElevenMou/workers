"""Supervisor loop that manages all worker subprocesses."""

import multiprocessing
import os
import re
import signal
import socket
import sys
import time
import uuid

_shutdown_requested = False


def _sigterm_handler(signum: int, frame: object) -> None:
    """Translate SIGTERM into the same shutdown path as KeyboardInterrupt."""
    global _shutdown_requested
    _shutdown_requested = True
    raise KeyboardInterrupt

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
SOCIAL_PRIORITY_QUEUE = "social-publishing-priority"
SOCIAL_QUEUE = "social-publishing"
VIDEO_QUEUE_ORDER = [VIDEO_PRIORITY_QUEUE, VIDEO_QUEUE]
CLIP_QUEUE_ORDER = [CLIP_PRIORITY_QUEUE, CLIP_QUEUE]
SOCIAL_QUEUE_ORDER = [SOCIAL_PRIORITY_QUEUE, SOCIAL_QUEUE]
VIDEO_PREFIX = "video-worker"
CLIP_PREFIX = "clip-worker"
SOCIAL_PREFIX = "social-worker"
MAINTENANCE_LEADER_KEY = "clipry:maintenance:leader"

_CRASH_WINDOW_SECONDS = 120
_CRASH_THRESHOLD = 3
_BASE_CRASH_BACKOFF_SECONDS = 5
_MAX_CRASH_BACKOFF_SECONDS = 60


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


def _cleanup_failed_job_registries(
    conn,
    queue_names: list[str],
    max_age_seconds: int = 86400 * 7,  # 7 days
) -> int:
    """Remove failed jobs older than max_age_seconds from all queue registries.

    Before deletion, permanently failed jobs are recorded in the dead letter
    queue for later inspection and alerting.
    """
    from rq import Queue
    from rq.job import Job
    from utils.dead_letter_queue import record_dead_letter

    cleaned = 0
    for queue_name in queue_names:
        try:
            queue = Queue(queue_name, connection=conn)
            registry = queue.failed_job_registry
            job_ids = registry.get_job_ids()
            for job_id in job_ids:
                try:
                    job = Job.fetch(job_id, connection=conn)
                    if job.ended_at:
                        age = time.time() - job.ended_at.timestamp()
                        if age > max_age_seconds:
                            record_dead_letter(
                                conn,
                                job_id=job_id,
                                queue_name=queue_name,
                                task_path=job.func_name or "unknown",
                                error_message=str(job.exc_info or "Unknown error"),
                                job_data=job.args[0] if job.args else None,
                                attempt_count=job.retries_left if hasattr(job, "retries_left") else 0,
                            )
                            registry.remove(job)
                            job.delete()
                            cleaned += 1
                except Exception:
                    try:
                        registry.remove(job_id)
                        cleaned += 1
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Failed to clean failed registry for %s: %s", queue_name, exc)

    if cleaned:
        logger.info("Cleaned %d stale entries from failed job registries (recorded to DLQ)", cleaned)
    return cleaned


def _run_maintenance_loop(
    *,
    startup_conn,
    queue_names: list[str],
    raw_video_cleanup_interval_seconds: int,
    clip_asset_cleanup_interval_seconds: int,
    processing_job_stale_seconds: int,
    leader_lock_ttl_seconds: int,
    leader_renew_seconds: int,
    social_publication_dispatch_interval_seconds: int,
    social_publication_claim_batch_size: int,
) -> int:
    from tasks.clips.cleanup import cleanup_expired_clip_assets
    from tasks.videos.cleanup import cleanup_expired_raw_videos
    from utils.redis_client import QueueFullError, enqueue_job
    from utils.supabase_client import assert_response_ok, supabase

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
    last_admission_reconciliation = 0.0
    admission_reconcile_interval = 300.0  # 5 minutes
    last_failed_job_cleanup = 0.0
    last_social_publication_dispatch = 0.0
    failed_job_cleanup_interval = 86400.0  # daily
    last_lock_renewal = 0.0
    renew_every = max(1, int(leader_renew_seconds))
    has_logged_standby = False

    signal.signal(signal.SIGTERM, _sigterm_handler)

    logger.info(
        "Supervisor role=maintenance started (leader_key=%s ttl=%ss renew=%ss)",
        MAINTENANCE_LEADER_KEY,
        max(5, int(leader_lock_ttl_seconds)),
        renew_every,
    )
    should_warm_whisper = env_bool("MAINTENANCE_WARM_WHISPER", False)
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
                social_publication_dispatch_interval_seconds > 0
                and now - last_social_publication_dispatch >= social_publication_dispatch_interval_seconds
            ):
                try:
                    claimed_resp = supabase.rpc(
                        "claim_due_clip_publications",
                        {"p_limit": int(social_publication_claim_batch_size)},
                    ).execute()
                    assert_response_ok(
                        claimed_resp,
                        "Failed to claim due social publications",
                    )
                    claimed_rows = list(claimed_resp.data or [])
                    for publication in claimed_rows:
                        batch_resp = (
                            supabase.table("clip_publish_batches")
                            .select("id, user_id, team_id, billing_owner_user_id")
                            .eq("id", publication["batch_id"])
                            .single()
                            .execute()
                        )
                        assert_response_ok(
                            batch_resp,
                            f"Failed to load publish batch {publication['batch_id']}",
                        )
                        batch = batch_resp.data

                        sub_resp = (
                            supabase.table("subscriptions")
                            .select("tier, status, interval")
                            .eq("user_id", batch["billing_owner_user_id"])
                            .limit(1)
                            .execute()
                        )
                        assert_response_ok(
                            sub_resp,
                            f"Failed to load subscription for {batch['billing_owner_user_id']}",
                        )
                        subscription_rows = sub_resp.data or []
                        sub_row = subscription_rows[0] if subscription_rows else {}
                        interval = str(sub_row.get("interval") or "month")
                        tier = str(sub_row.get("tier") or "free")

                        plan_resp = (
                            supabase.table("pricing_tiers")
                            .select("priority_processing")
                            .eq("tier", tier)
                            .eq("interval", interval)
                            .eq("active", True)
                            .limit(1)
                            .execute()
                        )
                        assert_response_ok(
                            plan_resp,
                            f"Failed to load pricing tier for {batch['billing_owner_user_id']}",
                        )
                        plan_rows = plan_resp.data or []
                        priority_processing = bool(
                            plan_rows[0].get("priority_processing") if plan_rows else False
                        )

                        queue_name = (
                            SOCIAL_PRIORITY_QUEUE if priority_processing else SOCIAL_QUEUE
                        )
                        job_id = str(uuid.uuid4())
                        job_data = {
                            "jobId": job_id,
                            "publicationId": publication["id"],
                            "clipId": publication["clip_id"],
                            "userId": batch["user_id"],
                            "workspaceTeamId": batch.get("team_id"),
                            "billingOwnerUserId": batch["billing_owner_user_id"],
                            "provider": publication["provider"],
                            "subscriptionTier": tier,
                        }
                        job_resp = (
                            supabase.table("jobs")
                            .insert(
                                {
                                    "id": job_id,
                                    "user_id": batch["user_id"],
                                    "team_id": batch.get("team_id"),
                                    "billing_owner_user_id": batch["billing_owner_user_id"],
                                    "clip_id": publication["clip_id"],
                                    "publication_id": publication["id"],
                                    "type": "publish_clip",
                                    "status": "queued",
                                    "input_data": job_data,
                                }
                            )
                            .execute()
                        )
                        assert_response_ok(job_resp, f"Failed to insert publish job {job_id}")
                        try:
                            enqueue_job(queue_name, "tasks.publishing.publish.publish_clip_task", job_data, job_id=job_id)
                        except QueueFullError as exc:
                            logger.warning(
                                "Social queue full while dispatching publication %s: %s",
                                publication["id"],
                                exc,
                            )
                            supabase.table("jobs").delete().eq("id", job_id).execute()
                            supabase.table("clip_publications").update(
                                {
                                    "status": "scheduled",
                                    "queued_at": None,
                                    "last_error": str(exc),
                                }
                            ).eq("id", publication["id"]).execute()
                        except Exception as exc:
                            logger.warning(
                                "Social publication dispatch failed for %s: %s",
                                publication["id"],
                                exc,
                            )
                            supabase.table("jobs").update(
                                {"status": "failed", "error_message": str(exc)}
                            ).eq("id", job_id).execute()
                            supabase.table("clip_publications").update(
                                {
                                    "status": "scheduled",
                                    "queued_at": None,
                                    "last_error": str(exc),
                                }
                            ).eq("id", publication["id"]).execute()
                except Exception as exc:
                    logger.warning("Scheduled social publication tick failed: %s", exc)
                finally:
                    last_social_publication_dispatch = now

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

            # Periodic admission counter reconciliation (every 5 min)
            if now - last_admission_reconciliation >= admission_reconcile_interval:
                try:
                    from utils.redis_client import reconcile_admission_counts
                    reconciled = reconcile_admission_counts(startup_conn)
                    logger.debug(
                        "Periodic admission reconciliation: video=%s clip=%s social=%s",
                        reconciled.get("video", 0),
                        reconciled.get("clip", 0),
                        reconciled.get("social", 0),
                    )
                except Exception as exc:
                    logger.warning("Periodic admission reconciliation failed: %s", exc)
                finally:
                    last_admission_reconciliation = now

            # Periodic failed job registry cleanup (daily)
            if now - last_failed_job_cleanup >= failed_job_cleanup_interval:
                try:
                    _cleanup_failed_job_registries(startup_conn, queue_names)
                except Exception as exc:
                    logger.warning("Failed job cleanup tick failed: %s", exc)
                finally:
                    last_failed_job_cleanup = now

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
        NUM_SOCIAL_WORKERS,
        NUM_VIDEO_WORKERS,
        PROCESSING_JOB_STALE_SECONDS,
        RAW_VIDEO_CLEANUP_INTERVAL_SECONDS,
        SOCIAL_PUBLICATION_CLAIM_BATCH_SIZE,
        SOCIAL_PUBLICATION_DISPATCH_INTERVAL_SECONDS,
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
    from utils.minio_client import initialize_minio_storage

    validate_env()
    initialize_minio_storage()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    logger.info("SIGTERM handler registered for graceful Docker shutdown")

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
        SOCIAL_PRIORITY_QUEUE,
        SOCIAL_QUEUE,
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
            social_publication_dispatch_interval_seconds=SOCIAL_PUBLICATION_DISPATCH_INTERVAL_SECONDS,
            social_publication_claim_batch_size=SOCIAL_PUBLICATION_CLAIM_BATCH_SIZE,
        )

    try:
        reconciled = reconcile_admission_counts(startup_conn)
        logger.info(
            "Reconciled admission counters at startup: video=%d clip=%d social=%d",
            int(reconciled.get("video", 0)),
            int(reconciled.get("clip", 0)),
            int(reconciled.get("social", 0)),
        )
    except Exception as exc:
        logger.warning("Admission-counter reconciliation failed at startup: %s", exc)

    # Optional WORKER_MODE restricts this supervisor to one worker group.
    worker_mode = (os.getenv("WORKER_MODE") or "").strip().lower()
    if worker_mode == "video":
        effective_num_video = NUM_VIDEO_WORKERS
        effective_num_clip = 0
        effective_num_social = 0
        logger.info("WORKER_MODE=video - only video workers will be managed")
    elif worker_mode == "clip":
        effective_num_video = 0
        effective_num_clip = NUM_CLIP_WORKERS
        effective_num_social = 0
        logger.info("WORKER_MODE=clip - only clip workers will be managed")
    elif worker_mode == "social":
        effective_num_video = 0
        effective_num_clip = 0
        effective_num_social = NUM_SOCIAL_WORKERS
        logger.info("WORKER_MODE=social - only social workers will be managed")
    else:
        effective_num_video = NUM_VIDEO_WORKERS
        effective_num_clip = NUM_CLIP_WORKERS
        effective_num_social = NUM_SOCIAL_WORKERS

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
            desired_social_workers = 0
        elif worker_mode == "clip":
            desired_video_workers = 0
            desired_clip_workers = set_group_worker_scale_target(
                group="clip",
                workers=effective_num_clip,
                connection=startup_conn,
                default=effective_num_clip,
            )
            desired_social_workers = 0
        elif worker_mode == "social":
            desired_video_workers = 0
            desired_clip_workers = 0
            desired_social_workers = set_group_worker_scale_target(
                group="social",
                workers=effective_num_social,
                connection=startup_conn,
                default=effective_num_social,
            )
        else:
            desired_video_workers, desired_clip_workers, desired_social_workers = set_worker_scale_target(
                connection=startup_conn,
                video_workers=effective_num_video,
                clip_workers=effective_num_clip,
                social_workers=effective_num_social,
                default_video=effective_num_video,
                default_clip=effective_num_clip,
                default_social=effective_num_social,
            )
        logger.info(
            "Reset worker scale target on startup to defaults: video=%d clip=%d social=%d",
            desired_video_workers,
            desired_clip_workers,
            desired_social_workers,
        )
    else:
        if worker_mode == "video":
            desired_video_workers = get_group_worker_scale_target(
                group="video",
                connection=startup_conn,
                default=effective_num_video,
            )
            desired_clip_workers = 0
            desired_social_workers = 0
        elif worker_mode == "clip":
            desired_video_workers = 0
            desired_clip_workers = get_group_worker_scale_target(
                group="clip",
                connection=startup_conn,
                default=effective_num_clip,
            )
            desired_social_workers = 0
        elif worker_mode == "social":
            desired_video_workers = 0
            desired_clip_workers = 0
            desired_social_workers = get_group_worker_scale_target(
                group="social",
                connection=startup_conn,
                default=effective_num_social,
            )
        else:
            desired_video_workers, desired_clip_workers, desired_social_workers = get_worker_scale_target(
                connection=startup_conn,
                default_video=effective_num_video,
                default_clip=effective_num_clip,
                default_social=effective_num_social,
            )

    # Enforce WORKER_MODE limits even when reading from Redis.
    if worker_mode == "video":
        desired_clip_workers = 0
        desired_social_workers = 0
    elif worker_mode == "clip":
        desired_video_workers = 0
        desired_social_workers = 0
    elif worker_mode == "social":
        desired_video_workers = 0
        desired_clip_workers = 0

    for i in range(desired_video_workers):
        worker_specs[f"{VIDEO_PREFIX}-{instance_id}-{i}"] = list(VIDEO_QUEUE_ORDER)
    for i in range(desired_clip_workers):
        worker_specs[f"{CLIP_PREFIX}-{instance_id}-{i}"] = list(CLIP_QUEUE_ORDER)
    for i in range(desired_social_workers):
        worker_specs[f"{SOCIAL_PREFIX}-{instance_id}-{i}"] = list(SOCIAL_QUEUE_ORDER)

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
    for i in range(desired_social_workers):
        name = f"{SOCIAL_PREFIX}-{instance_id}-{i}"
        p = spawn_worker(worker_specs[name], name)
        processes[name] = p
        logger.info("Spawned %s  (pid %d)", name, p.pid)

    current_video_workers = desired_video_workers
    current_clip_workers = desired_clip_workers
    current_social_workers = desired_social_workers
    crash_timestamps: dict[str, list[float]] = {}

    total = current_video_workers + current_clip_workers + current_social_workers
    logger.info(
        "Worker pool ready - instance=%s %d video, %d clip, %d social (%d total)",
        instance_id,
        current_video_workers,
        current_clip_workers,
        current_social_workers,
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
                    target_social_workers = 0
                elif worker_mode == "clip":
                    target_video_workers = 0
                    target_clip_workers = get_group_worker_scale_target(
                        group="clip",
                        connection=startup_conn,
                        default=current_clip_workers,
                    )
                    target_social_workers = 0
                elif worker_mode == "social":
                    target_video_workers = 0
                    target_clip_workers = 0
                    target_social_workers = get_group_worker_scale_target(
                        group="social",
                        connection=startup_conn,
                        default=current_social_workers,
                    )
                else:
                    target_video_workers, target_clip_workers, target_social_workers = get_worker_scale_target(
                        connection=startup_conn,
                        default_video=current_video_workers,
                        default_clip=current_clip_workers,
                        default_social=current_social_workers,
                    )
            except Exception as exc:
                logger.warning("Failed to read scale target from Redis: %s", exc)
                target_video_workers = current_video_workers
                target_clip_workers = current_clip_workers
                target_social_workers = current_social_workers

            if (
                target_video_workers != current_video_workers
                or target_clip_workers != current_clip_workers
                or target_social_workers != current_social_workers
            ):
                logger.info(
                    "Applying scale target - video: %d -> %d, clip: %d -> %d, social: %d -> %d",
                    current_video_workers,
                    target_video_workers,
                    current_clip_workers,
                    target_clip_workers,
                    current_social_workers,
                    target_social_workers,
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
                current_social_workers = resize_group(
                    conn=startup_conn,
                    worker_specs=worker_specs,
                    processes=processes,
                    prefix=f"{SOCIAL_PREFIX}-{instance_id}",
                    queue_names=list(SOCIAL_QUEUE_ORDER),
                    current_count=current_social_workers,
                    target_count=target_social_workers,
                )

            for name, p in list(processes.items()):
                if not p.is_alive():
                    exit_code = p.exitcode
                    now_mono = time.monotonic()
                    crash_ts = crash_timestamps.get(name, [])
                    crash_ts = [t for t in crash_ts if now_mono - t < _CRASH_WINDOW_SECONDS]
                    crash_ts.append(now_mono)
                    crash_timestamps[name] = crash_ts

                    if len(crash_ts) >= _CRASH_THRESHOLD:
                        backoff = min(
                            _MAX_CRASH_BACKOFF_SECONDS,
                            _BASE_CRASH_BACKOFF_SECONDS * (2 ** (len(crash_ts) - _CRASH_THRESHOLD)),
                        )
                        logger.warning(
                            "Worker %s crashed %d times in %ds (exit code %s). "
                            "Backing off %.0fs before restart.",
                            name,
                            len(crash_ts),
                            _CRASH_WINDOW_SECONDS,
                            exit_code,
                            backoff,
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning(
                            "Worker %s exited unexpectedly (exit code %s). Restarting.",
                            name,
                            exit_code,
                        )

                    cleanup_named_workers(startup_conn, {name})
                    replacement = spawn_worker(worker_specs[name], name)
                    processes[name] = replacement
                    logger.info("Respawned %s (pid %d)", name, replacement.pid)
                    continue

                # Check heartbeat: detect alive-but-hung workers
                from worker_supervisor.heartbeat import get_heartbeat_age
                hb_age = get_heartbeat_age(startup_conn, name)
                if hb_age is not None and hb_age > 120:
                    logger.warning(
                        "Worker %s heartbeat stale (%.0fs). Killing and restarting.",
                        name,
                        hb_age,
                    )
                    p.kill()
                    p.join(timeout=5)
                    cleanup_named_workers(startup_conn, {name})
                    replacement = spawn_worker(worker_specs[name], name)
                    processes[name] = replacement
                    logger.info("Respawned hung worker %s (pid %d)", name, replacement.pid)

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down workers ...")

        for name, p in list(processes.items()):
            stop_worker_process(startup_conn, name, p)

        cleanup_named_workers(startup_conn, set(worker_specs.keys()))
        logger.info("All workers stopped.")
        return 0
    except Exception as exc:
        logger.exception("Supervisor crashed unexpectedly")
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(exc)
        except ImportError:
            pass
        return 1


def main() -> int:
    """CLI entrypoint helper."""
    return run_supervisor()


if __name__ == "__main__":
    sys.exit(main())

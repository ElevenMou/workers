"""Health endpoint router."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from api_app.auth import AuthenticatedUser, require_admin_user
from config import REDIS_CONNECTION_ALERT_THRESHOLD
from utils.redis_client import (
    QUEUE_CONFIG,
    get_admission_counts,
    get_queue_reject_counts,
    get_redis_connection,
    get_worker_scale_target,
)
from utils.supabase_client import assert_response_ok, supabase

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * percentile))
    return float(sorted_values[max(0, min(idx, len(sorted_values) - 1))])


def _queue_age_stats(conn, queue_name: str, *, sample_size: int = 500) -> dict[str, float | int]:
    now = datetime.now(timezone.utc)
    key = f"rq:queue:{queue_name}"
    try:
        job_ids = conn.lrange(key, 0, max(0, int(sample_size) - 1))
    except Exception:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    ages: list[float] = []
    for raw_job_id in job_ids:
        job_id = raw_job_id.decode("utf-8", errors="ignore") if isinstance(raw_job_id, bytes) else str(raw_job_id)
        if not job_id:
            continue
        try:
            payload = conn.hgetall(f"rq:job:{job_id}") or {}
        except Exception:
            continue
        enqueued_at = payload.get(b"enqueued_at") or payload.get("enqueued_at")
        created_at = payload.get(b"created_at") or payload.get("created_at")
        dt = _parse_timestamp(enqueued_at) or _parse_timestamp(created_at)
        if dt is None:
            continue
        ages.append(max(0.0, (now - dt).total_seconds()))

    if not ages:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(ages),
        "p50": round(_percentile(ages, 0.50), 3),
        "p95": round(_percentile(ages, 0.95), 3),
        "max": round(max(ages), 3),
    }


def _job_duration_stats(sample_size: int = 2000) -> dict[str, float | int]:
    try:
        response = (
            supabase.table("jobs")
            .select("status,started_at,completed_at")
            .order("created_at", desc=True)
            .limit(sample_size)
            .execute()
        )
        assert_response_ok(response, "Failed to query jobs for duration stats")
    except Exception:
        return {
            "completed_count": 0,
            "failed_count": 0,
            "retrying_count": 0,
            "duration_p50": 0.0,
            "duration_p95": 0.0,
        }

    completed_count = 0
    failed_count = 0
    retrying_count = 0
    durations: list[float] = []
    for row in response.data or []:
        status = str(row.get("status") or "").strip().lower()
        if status == "completed":
            completed_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "retrying":
            retrying_count += 1

        started_at = _parse_timestamp(row.get("started_at"))
        completed_at = _parse_timestamp(row.get("completed_at"))
        if started_at is None or completed_at is None:
            continue
        durations.append(max(0.0, (completed_at - started_at).total_seconds()))

    return {
        "completed_count": completed_count,
        "failed_count": failed_count,
        "retrying_count": retrying_count,
        "duration_p50": round(_percentile(durations, 0.50), 3) if durations else 0.0,
        "duration_p95": round(_percentile(durations, 0.95), 3) if durations else 0.0,
    }


def _stale_recovery_counts(conn) -> dict[str, int]:
    def _as_int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "total": _as_int(conn.get("clipry:metrics:stale_recoveries:total")),
        "clips": _as_int(conn.get("clipry:metrics:stale_recoveries:clips")),
        "videos": _as_int(conn.get("clipry:metrics:stale_recoveries:videos")),
    }


def _billing_dedupe_counts(conn) -> dict[str, int]:
    def _as_int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "total": _as_int(conn.get("clipry:metrics:billing_dedupe:total")),
        "owner_wallet": _as_int(conn.get("clipry:metrics:billing_dedupe:owner_wallet")),
        "team_wallet": _as_int(conn.get("clipry:metrics:billing_dedupe:team_wallet")),
    }


def _source_lock_wait_stats(sample_size: int = 500) -> dict[str, float | int]:
    try:
        response = (
            supabase.table("jobs")
            .select("result_data")
            .eq("type", "generate_clip")
            .order("created_at", desc=True)
            .limit(sample_size)
            .execute()
        )
        assert_response_ok(response, "Failed to query source-lock wait stats")
    except Exception:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    waits: list[float] = []
    for row in response.data or []:
        result_data = row.get("result_data")
        if not isinstance(result_data, dict):
            continue
        value = result_data.get("source_video_wait_seconds")
        try:
            wait_seconds = float(value)
        except (TypeError, ValueError):
            continue
        waits.append(max(0.0, wait_seconds))

    if not waits:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(waits),
        "p50": round(_percentile(waits, 0.50), 3),
        "p95": round(_percentile(waits, 0.95), 3),
        "max": round(max(waits), 3),
    }


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    checks = {"redis": "unknown", "supabase": "unknown"}

    try:
        get_redis_connection().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.error("Redis readiness check failed: %s", exc)
        checks["redis"] = "error"

    try:
        resp = supabase.table("jobs").select("id").limit(1).execute()
        assert_response_ok(resp, "Supabase readiness check failed")
        checks["supabase"] = "ok"
    except Exception as exc:
        logger.error("Supabase readiness check failed: %s", exc)
        checks["supabase"] = "error"

    if checks["redis"] != "ok" or checks["supabase"] != "ok":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}


@router.get("/health/metrics")
def health_metrics(
    _: AuthenticatedUser = Depends(require_admin_user),
) -> dict:
    """Return queue depths, worker scale targets, and Redis memory stats."""
    conn = get_redis_connection()

    queues: dict[str, int] = {}
    for queue_name in QUEUE_CONFIG:
        try:
            queues[queue_name] = int(conn.llen(f"rq:queue:{queue_name}"))
        except Exception:
            queues[queue_name] = -1

    workers = {"video_workers": -1, "clip_workers": -1, "social_workers": -1}
    try:
        video_workers, clip_workers, social_workers = get_worker_scale_target(
            connection=conn,
            default_video=0,
            default_clip=0,
            default_social=0,
        )
        workers = {
            "video_workers": int(video_workers),
            "clip_workers": int(clip_workers),
            "social_workers": int(social_workers),
        }
    except Exception:
        pass

    queue_rejects = get_queue_reject_counts(conn)
    admissions = get_admission_counts(conn)
    queue_age_seconds = {
        queue_name: _queue_age_stats(conn, queue_name)
        for queue_name in QUEUE_CONFIG
    }

    redis_info: dict[str, object] = {}
    try:
        mem = conn.info("memory")
        redis_info["used_memory_human"] = mem.get("used_memory_human", "unknown")
        redis_info["used_memory_peak_human"] = mem.get("used_memory_peak_human", "unknown")
        redis_info["maxmemory_human"] = mem.get("maxmemory_human", "unknown")
    except Exception:
        pass

    try:
        clients = conn.info("clients")
        connected_clients = int(clients.get("connected_clients") or 0)
        max_clients = int(clients.get("maxclients") or 0)
        redis_info["connected_clients"] = connected_clients
        redis_info["max_clients"] = max_clients
        if max_clients > 0:
            usage_ratio = connected_clients / max_clients
            redis_info["connection_usage_ratio"] = round(usage_ratio, 4)
            redis_info["connection_alert"] = usage_ratio >= REDIS_CONNECTION_ALERT_THRESHOLD
    except Exception:
        pass

    from utils.dead_letter_queue import get_dlq_count

    return {
        "queues": queues,
        "queue_age_seconds": queue_age_seconds,
        "queue_rejects": queue_rejects,
        "admissions": admissions,
        "stale_recovery": _stale_recovery_counts(conn),
        "source_lock_wait_seconds": _source_lock_wait_stats(),
        "billing_dedupe": _billing_dedupe_counts(conn),
        "dead_letter_queue": {
            "total_events": get_dlq_count(conn),
        },
        "workers": workers,
        "jobs": _job_duration_stats(),
        "redis": redis_info,
    }

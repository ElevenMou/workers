"""Health endpoint router."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api_app.auth import AuthenticatedUser, require_admin_user
from config import REDIS_CONNECTION_ALERT_THRESHOLD
from utils.redis_client import (
    QUEUE_CONFIG,
    get_queue_reject_counts,
    get_redis_connection,
    get_worker_scale_target,
)
from utils.supabase_client import assert_response_ok, supabase

logger = logging.getLogger(__name__)

router = APIRouter()


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

    workers = {"video_workers": -1, "clip_workers": -1}
    try:
        video_workers, clip_workers = get_worker_scale_target(
            connection=conn,
            default_video=0,
            default_clip=0,
        )
        workers = {
            "video_workers": int(video_workers),
            "clip_workers": int(clip_workers),
        }
    except Exception:
        pass

    queue_rejects = get_queue_reject_counts(conn)

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

    return {
        "queues": queues,
        "queue_rejects": queue_rejects,
        "workers": workers,
        "redis": redis_info,
    }


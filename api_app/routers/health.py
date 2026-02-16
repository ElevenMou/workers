"""Health endpoint router."""

from fastapi import APIRouter, HTTPException, status

from utils.redis_client import get_redis_connection
from utils.supabase_client import assert_response_ok, supabase

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
        checks["redis"] = f"error: {exc}"

    try:
        resp = supabase.table("jobs").select("id").limit(1).execute()
        assert_response_ok(resp, "Supabase readiness check failed")
        checks["supabase"] = "ok"
    except Exception as exc:
        checks["supabase"] = f"error: {exc}"

    if checks["redis"] != "ok" or checks["supabase"] != "ok":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}

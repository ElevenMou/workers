"""Shared helper functions for API routers."""

from fastapi import HTTPException, status

from api_app.state import logger
from utils.redis_client import enqueue_job
from utils.supabase_client import assert_response_ok, supabase


def _map_supabase_error_to_http_status(error_text: str) -> int:
    lowered = (error_text or "").lower()

    if "duplicate key value violates unique constraint" in lowered:
        return status.HTTP_409_CONFLICT

    if "violates foreign key constraint" in lowered:
        return status.HTTP_400_BAD_REQUEST

    if (
        "violates check constraint" in lowered
        or "invalid input syntax" in lowered
        or "invalid uuid" in lowered
        or "malformed uuid" in lowered
    ):
        return status.HTTP_400_BAD_REQUEST

    return status.HTTP_500_INTERNAL_SERVER_ERROR


def raise_on_error(response, context: str):
    """Map Supabase operation errors to suitable HTTP responses."""
    try:
        assert_response_ok(response, context)
    except RuntimeError as exc:
        detail = str(exc)
        raise HTTPException(
            status_code=_map_supabase_error_to_http_status(detail),
            detail=detail,
        ) from exc


def enqueue_or_fail(
    *,
    queue_name: str,
    task_path: str,
    job_data: dict,
    job_id: str,
    user_id: str,
    job_type: str,
    video_id: str,
    clip_id: str | None = None,
):
    """Enqueue a background job and mark DB job failed if enqueue fails."""
    try:
        enqueue_job(queue_name, task_path, job_data, job_id=job_id)
    except Exception as exc:
        err = f"Queue enqueue failed: {exc}"
        logger.error("Failed to enqueue job %s: %s", job_id, exc)

        fail_resp = (
            supabase.table("jobs")
            .update({"status": "failed", "error_message": err})
            .eq("id", job_id)
            .execute()
        )
        raise_on_error(fail_resp, "Failed to mark job as failed after enqueue error")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to queue job right now. Please retry.",
        ) from exc

    logger.info(
        "Job %s enqueued on %s queue (type=%s user=%s video=%s clip=%s)",
        job_id,
        queue_name,
        job_type,
        user_id,
        video_id,
        clip_id,
    )

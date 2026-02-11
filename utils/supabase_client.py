import logging
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, validate_env

logger = logging.getLogger(__name__)

# Monkey-patch gotrue/httpx proxy argument mismatch (httpx.Client expects "proxies")
try:
    import httpx
    from gotrue._sync import gotrue_base_api as _gotrue_base_api
    from gotrue import http_clients as _gotrue_http_clients

    class _PatchedSyncClient(httpx.Client):
        def __init__(self, *args, proxy=None, **kwargs):
            if proxy is not None and "proxies" not in kwargs:
                kwargs["proxies"] = proxy
            super().__init__(*args, **kwargs)

        def aclose(self) -> None:
            self.close()

    _gotrue_http_clients.SyncClient = _PatchedSyncClient  # type: ignore[attr-defined]

    # Override GoTrue base API to avoid passing unsupported "proxy" kw to httpx.Client
    def _patched_gotrue_init(self, *, url, headers, http_client, verify=True, proxy=None):
        self._url = url
        self._headers = headers
        self._http_client = http_client or _PatchedSyncClient(
            verify=bool(verify),
            proxies=proxy if proxy is not None else None,
            follow_redirects=True,
            http2=True,
        )

    _gotrue_base_api.SyncGoTrueBaseAPI.__init__ = _patched_gotrue_init  # type: ignore[assignment]
except Exception:
    # If patch fails, let the original error surface; this keeps startup resilient.
    pass

validate_env()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _error_detail(error: Any) -> str:
    if error is None:
        return "unknown error"
    return getattr(error, "message", None) or str(error)


def assert_response_ok(response: Any, context: str):
    """Raise RuntimeError when Supabase returns an error payload."""
    error = getattr(response, "error", None)
    if error:
        detail = _error_detail(error)
        raise RuntimeError(f"{context}: {detail}")
    return response


def _charge_credits_or_raise(
    *,
    user_id: str,
    amount: int,
    tx_type: str,
    description: str,
    video_id: str | None = None,
    clip_id: str | None = None,
    context: str,
):
    """Atomically charge credits in DB and fail when balance is insufficient."""
    resp = supabase.rpc(
        "charge_credits",
        {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_type": tx_type,
            "p_description": description,
            "p_video_id": video_id,
            "p_clip_id": clip_id,
        },
    ).execute()
    assert_response_ok(resp, context)
    if not resp.data:
        raise RuntimeError(f"{context}: insufficient credits")


def charge_clip_generation_credits(
    *,
    user_id: str,
    amount: int,
    description: str,
    video_id: str,
    clip_id: str,
):
    _charge_credits_or_raise(
        user_id=user_id,
        amount=amount,
        tx_type="clip_generation",
        description=description,
        video_id=video_id,
        clip_id=clip_id,
        context="Failed to charge clip-generation credits",
    )


def charge_video_analysis_credits(
    *,
    user_id: str,
    amount: int,
    description: str,
    video_id: str,
):
    _charge_credits_or_raise(
        user_id=user_id,
        amount=amount,
        tx_type="video_analysis",
        description=description,
        video_id=video_id,
        clip_id=None,
        context="Failed to charge video-analysis credits",
    )


def update_job_status(
    job_id: str,
    status: str,
    progress: int,
    error: str = None,
    result_data: dict = None,
):
    """Update job status and progress"""
    data = {"status": status, "progress": progress}
    now_utc = datetime.now(timezone.utc).isoformat()

    if status == "processing" and progress == 0:
        data["started_at"] = now_utc
    elif status in ["completed", "failed"]:
        data["completed_at"] = now_utc

    if error:
        data["error_message"] = error

    if result_data:
        data["result_data"] = result_data

    resp = supabase.table("jobs").update(data).eq("id", job_id).execute()
    assert_response_ok(resp, f"Failed to update job status for {job_id}")


def update_video_status(video_id: str, status: str, **kwargs):
    """Update video record"""
    data = {"status": status}
    data.update(kwargs)
    resp = supabase.table("videos").update(data).eq("id", video_id).execute()
    assert_response_ok(resp, f"Failed to update video status for {video_id}")

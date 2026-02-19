import logging
import os
import random
import ssl
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from supabase import Client, create_client

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, validate_env

logger = logging.getLogger(__name__)

_RETRYABLE_METHODS = {"GET", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"}
_DEFAULT_MAX_ATTEMPTS = 4
_DEFAULT_BASE_DELAY_SECONDS = 0.2
_DEFAULT_MAX_DELAY_SECONDS = 2.0
_DEFAULT_JITTER_SECONDS = 0.1


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %d",
            name,
            value,
            default,
        )
        return default
    return max(1, parsed)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %.2f",
            name,
            value,
            default,
        )
        return default
    return max(0.0, parsed)


SUPABASE_HTTP_MAX_ATTEMPTS = _env_int(
    "SUPABASE_HTTP_MAX_ATTEMPTS",
    _DEFAULT_MAX_ATTEMPTS,
)
SUPABASE_HTTP_RETRY_BASE_DELAY_SECONDS = _env_float(
    "SUPABASE_HTTP_RETRY_BASE_DELAY_SECONDS",
    _DEFAULT_BASE_DELAY_SECONDS,
)
SUPABASE_HTTP_RETRY_MAX_DELAY_SECONDS = _env_float(
    "SUPABASE_HTTP_RETRY_MAX_DELAY_SECONDS",
    _DEFAULT_MAX_DELAY_SECONDS,
)
SUPABASE_HTTP_RETRY_JITTER_SECONDS = _env_float(
    "SUPABASE_HTTP_RETRY_JITTER_SECONDS",
    _DEFAULT_JITTER_SECONDS,
)


def _is_retryable_supabase_error(exc: Exception) -> bool:
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
            ),
        ):
            return True
    except Exception:
        # If httpx imports unexpectedly fail, be conservative and skip retries.
        return False

    cause = getattr(exc, "__cause__", None)
    return isinstance(cause, ssl.SSLError)


def _retry_sleep_seconds(attempt_index: int) -> float:
    backoff = SUPABASE_HTTP_RETRY_BASE_DELAY_SECONDS * (2 ** attempt_index)
    capped = min(backoff, SUPABASE_HTTP_RETRY_MAX_DELAY_SECONDS)
    jitter = random.uniform(0, SUPABASE_HTTP_RETRY_JITTER_SECONDS)
    return capped + jitter


def _patch_client_request_with_retry(client_cls: type, client_name: str) -> None:
    if getattr(client_cls, "_clipry_retry_patched", False):
        return

    original_request = client_cls.request

    @wraps(original_request)
    def _patched_request(self, method, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        method_upper = str(method).upper()
        if method_upper not in _RETRYABLE_METHODS or SUPABASE_HTTP_MAX_ATTEMPTS <= 1:
            return original_request(self, method, url, *args, **kwargs)

        last_exc: Exception | None = None
        for attempt in range(SUPABASE_HTTP_MAX_ATTEMPTS):
            try:
                return original_request(self, method, url, *args, **kwargs)
            except Exception as exc:  # pragma: no cover - network path
                if not _is_retryable_supabase_error(exc):
                    raise
                last_exc = exc
                if attempt + 1 >= SUPABASE_HTTP_MAX_ATTEMPTS:
                    raise

                sleep_seconds = _retry_sleep_seconds(attempt)
                logger.warning(
                    "Transient Supabase %s error via %s (%s). "
                    "Retrying in %.2fs (%d/%d) ...",
                    method_upper,
                    client_name,
                    exc,
                    sleep_seconds,
                    attempt + 1,
                    SUPABASE_HTTP_MAX_ATTEMPTS,
                )
                time.sleep(sleep_seconds)

        if last_exc:
            raise last_exc

        return original_request(self, method, url, *args, **kwargs)

    client_cls.request = _patched_request  # type: ignore[assignment]
    client_cls._clipry_retry_patched = True  # type: ignore[attr-defined]

# Monkey-patch gotrue/httpx proxy argument mismatch (httpx.Client expects "proxies")
try:
    import httpx
    from gotrue._sync import gotrue_base_api as _gotrue_base_api
    from gotrue import http_clients as _gotrue_http_clients
    import postgrest.utils as _postgrest_utils
    import storage3.utils as _storage_utils

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
    _patch_client_request_with_retry(_PatchedSyncClient, "gotrue")
    _patch_client_request_with_retry(_postgrest_utils.SyncClient, "postgrest")
    _patch_client_request_with_retry(_storage_utils.SyncClient, "storage")
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

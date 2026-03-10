import logging
import os
import random
import ssl
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from supabase import Client, create_client

from config import (
    POLAR_USAGE_EVENT_ANALYSIS_NAME,
    POLAR_USAGE_EVENT_GENERATION_NAME,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    validate_env,
)
from utils.polar_usage import emit_polar_usage_event

logger = logging.getLogger(__name__)
_FREE_RESET_RPC_AVAILABLE: bool | None = None

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

    _SUPABASE_HTTP_TIMEOUT = float(os.getenv("SUPABASE_HTTP_TIMEOUT_SECONDS", "30"))

    class _PatchedSyncClient(httpx.Client):
        def __init__(self, *args, proxy=None, **kwargs):
            if proxy is not None and "proxies" not in kwargs:
                kwargs["proxies"] = proxy
            # Set a default timeout so RPC calls don't hang workers indefinitely.
            kwargs.setdefault("timeout", _SUPABASE_HTTP_TIMEOUT)
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
except Exception as _patch_exc:
    # If patch fails, log a warning so operators know retry logic is inactive.
    logger.warning(
        "Failed to apply Supabase HTTP retry patches: %s. "
        "Retry logic will not be active.",
        _patch_exc,
    )

validate_env()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _error_detail(error: Any) -> str:
    if error is None:
        return "unknown error"
    return getattr(error, "message", None) or str(error)


def _is_missing_column_error(error: Any, column: str) -> bool:
    detail = _error_detail(error).lower()
    column_name = str(column or "").strip().lower()
    if not column_name:
        return False
    if column_name not in detail:
        return False
    return (
        "does not exist" in detail
        or "schema cache" in detail
        or "could not find" in detail
        or "not found" in detail
    )


def assert_response_ok(response: Any, context: str):
    """Raise RuntimeError when Supabase returns an error payload."""
    error = getattr(response, "error", None)
    if error:
        detail = _error_detail(error)
        raise RuntimeError(f"{context}: {detail}")
    return response


def _refresh_free_monthly_credits_if_needed(user_id: str):
    """Apply free-plan monthly reset (set balance to 20) when the DB function exists."""
    global _FREE_RESET_RPC_AVAILABLE

    if _FREE_RESET_RPC_AVAILABLE is False:
        return

    response = supabase.rpc(
        "reset_free_monthly_credits",
        {
            "p_user_id": user_id,
        },
    ).execute()

    error = getattr(response, "error", None)
    if not error:
        _FREE_RESET_RPC_AVAILABLE = True
        return

    code = getattr(error, "code", None)
    detail = _error_detail(error)
    is_missing_function = code == "42883" or "reset_free_monthly_credits" in detail
    if is_missing_function:
        if _FREE_RESET_RPC_AVAILABLE is not False:
            logger.warning(
                "Missing DB function reset_free_monthly_credits. "
                "Apply SQL/billing_reliability.sql to enable free-plan monthly resets."
            )
        _FREE_RESET_RPC_AVAILABLE = False
        return

    raise RuntimeError(f"Failed to refresh free monthly credits for {user_id}: {detail}")


def _charge_credits_or_raise(
    *,
    user_id: str,
    amount: int,
    tx_type: str,
    description: str,
    video_id: str | None = None,
    clip_id: str | None = None,
    processing_ref: str | None = None,
    context: str,
):
    """Atomically charge credits in DB and fail when balance is insufficient."""
    if processing_ref:
        existing_resp = (
            supabase.table("credit_transactions")
            .select("id")
            .eq("user_id", user_id)
            .eq("type", tx_type)
            .eq("processing_ref", processing_ref)
            .lt("amount", 0)
            .limit(1)
            .execute()
        )
        assert_response_ok(existing_resp, f"{context}: dedupe precheck failed")
        if existing_resp.data:
            try:
                from utils.redis_client import get_redis_connection

                conn = get_redis_connection()
                conn.incr("clipry:metrics:billing_dedupe:total")
                conn.incr("clipry:metrics:billing_dedupe:owner_wallet")
            except Exception:
                pass
            return

    _refresh_free_monthly_credits_if_needed(user_id)

    resp = supabase.rpc(
        "charge_credits",
        {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_type": tx_type,
            "p_description": description,
            "p_video_id": video_id,
            "p_clip_id": clip_id,
            "p_processing_ref": processing_ref,
        },
    ).execute()
    assert_response_ok(resp, context)
    if not resp.data:
        raise RuntimeError(f"{context}: insufficient credits")


def _charge_team_credits_or_raise(
    *,
    owner_user_id: str,
    team_id: str,
    amount: int,
    owner_tx_type: str,
    description: str,
    actor_user_id: str,
    video_id: str | None = None,
    clip_id: str | None = None,
    job_id: str | None = None,
    processing_ref: str | None = None,
    context: str,
):
    """Atomically charge a team wallet and write owner/team audit transactions."""
    if processing_ref:
        existing_resp = (
            supabase.table("team_wallet_transactions")
            .select("id")
            .eq("team_id", team_id)
            .eq("type", "processing_charge")
            .eq("processing_ref", processing_ref)
            .lt("amount", 0)
            .limit(1)
            .execute()
        )
        assert_response_ok(existing_resp, f"{context}: dedupe precheck failed")
        if existing_resp.data:
            try:
                from utils.redis_client import get_redis_connection

                conn = get_redis_connection()
                conn.incr("clipry:metrics:billing_dedupe:total")
                conn.incr("clipry:metrics:billing_dedupe:team_wallet")
            except Exception:
                pass
            return

    resp = supabase.rpc(
        "charge_team_credits",
        {
            "p_owner_user_id": owner_user_id,
            "p_team_id": team_id,
            "p_amount": amount,
            "p_owner_transaction_type": owner_tx_type,
            "p_description": description,
            "p_actor_user_id": actor_user_id,
            "p_video_id": video_id,
            "p_clip_id": clip_id,
            "p_job_id": job_id,
            "p_processing_ref": processing_ref,
        },
    ).execute()
    assert_response_ok(resp, context)
    if not resp.data:
        raise RuntimeError(f"{context}: insufficient team wallet credits")


def get_credit_balance(user_id: str) -> int:
    """Return current user credit balance (0 when missing)."""
    response = (
        supabase.table("credit_balances")
        .select("balance")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    assert_response_ok(response, f"Failed to load credit balance for {user_id}")

    rows = response.data or []
    if not rows:
        return 0

    try:
        return int(rows[0].get("balance") or 0)
    except (TypeError, ValueError):
        return 0


def get_team_wallet_balance(team_id: str) -> int:
    """Return team wallet balance (0 when missing)."""
    response = (
        supabase.table("team_wallets")
        .select("balance")
        .eq("team_id", team_id)
        .limit(1)
        .execute()
    )
    assert_response_ok(response, f"Failed to load team wallet balance for {team_id}")

    rows = response.data or []
    if not rows:
        return 0

    try:
        return int(rows[0].get("balance") or 0)
    except (TypeError, ValueError):
        return 0


def has_sufficient_credits(
    *,
    user_id: str,
    amount: int,
    charge_source: str = "owner_wallet",
    team_id: str | None = None,
) -> bool:
    """Check whether owner or team wallet has at least `amount` credits."""
    if amount <= 0:
        return True

    if charge_source == "team_wallet":
        if not team_id:
            return False
        response = supabase.rpc(
            "has_team_credits",
            {
                "p_team_id": team_id,
                "p_amount": amount,
            },
        ).execute()
        assert_response_ok(response, f"Failed to check team credits for {team_id}")
        return bool(response.data)

    _refresh_free_monthly_credits_if_needed(user_id)

    response = supabase.rpc(
        "has_credits",
        {
            "p_user_id": user_id,
            "p_amount": amount,
        },
    ).execute()
    assert_response_ok(response, f"Failed to check credits for {user_id}")

    return bool(response.data)


def reserve_credits(
    *,
    user_id: str,
    amount: int,
    reason: str,
    reservation_key: str | None = None,
    video_id: str | None = None,
    clip_id: str | None = None,
) -> str | None:
    """Reserve owner-wallet credits using an idempotency key."""
    parsed_amount = int(amount)
    if parsed_amount <= 0:
        return None

    _refresh_free_monthly_credits_if_needed(user_id)
    response = supabase.rpc(
        "reserve_credits",
        {
            "p_user_id": user_id,
            "p_amount": parsed_amount,
            "p_reason": reason,
            "p_reservation_key": reservation_key,
            "p_video_id": video_id,
            "p_clip_id": clip_id,
        },
    ).execute()
    assert_response_ok(response, f"Failed to reserve credits for {user_id}")
    reservation_id = response.data
    if reservation_id is None:
        return None
    return str(reservation_id)


def capture_credit_reservation(
    *,
    reservation_id: str,
    tx_type: str,
    description: str | None = None,
    processing_ref: str | None = None,
) -> bool:
    """Capture a previously reserved owner-wallet charge."""
    response = supabase.rpc(
        "capture_credit_reservation",
        {
            "p_reservation_id": reservation_id,
            "p_type": tx_type,
            "p_description": description,
            "p_processing_ref": processing_ref,
        },
    ).execute()
    assert_response_ok(
        response,
        f"Failed to capture credit reservation {reservation_id}",
    )
    return bool(response.data)


def release_credit_reservation(*, reservation_id: str) -> bool:
    """Release a previously reserved owner-wallet charge."""
    response = supabase.rpc(
        "release_credit_reservation",
        {"p_reservation_id": reservation_id},
    ).execute()
    assert_response_ok(
        response,
        f"Failed to release credit reservation {reservation_id}",
    )
    return bool(response.data)


def has_team_wallet_charge_for_job(
    *,
    team_id: str,
    job_id: str,
    owner_user_id: str | None = None,
) -> bool:
    """Return whether a team-wallet processing charge already exists for a job."""
    query = (
        supabase.table("team_wallet_transactions")
        .select("id")
        .eq("team_id", team_id)
        .eq("job_id", job_id)
        .eq("type", "processing_charge")
        .lt("amount", 0)
        .limit(1)
    )
    if owner_user_id:
        query = query.eq("owner_user_id", owner_user_id)

    response = query.execute()
    assert_response_ok(
        response,
        f"Failed to check existing team-wallet charge for job {job_id}",
    )
    return bool(response.data)


def _build_usage_external_id(
    *,
    category: str,
    owner_user_id: str,
    amount: int,
    job_id: str | None,
    video_id: str | None,
    clip_id: str | None,
) -> str:
    stable_ref = job_id or clip_id or video_id or owner_user_id
    return f"{category}:{stable_ref}:{owner_user_id}:{amount}"


def _emit_usage_event_after_charge(
    *,
    event_name: str,
    category: str,
    owner_user_id: str,
    actor_user_id: str,
    amount: int,
    charge_source: str,
    team_id: str | None,
    job_id: str | None,
    video_id: str | None,
    clip_id: str | None,
    usage_metadata: dict[str, Any] | None,
) -> None:
    external_id = _build_usage_external_id(
        category=category,
        owner_user_id=owner_user_id,
        amount=amount,
        job_id=job_id,
        video_id=video_id,
        clip_id=clip_id,
    )
    metadata: dict[str, Any] = {
        "units": int(amount),
        "transaction_type": category,
        "charge_source": charge_source,
        "team_id": team_id,
        "job_id": job_id,
        "video_id": video_id,
        "clip_id": clip_id,
        "billing_owner_user_id": owner_user_id,
        "actor_user_id": actor_user_id,
    }
    if usage_metadata:
        metadata.update(usage_metadata)

    emit_polar_usage_event(
        event_name=event_name,
        external_customer_id=owner_user_id,
        external_id=external_id,
        metadata=metadata,
    )


def charge_clip_generation_credits(
    *,
    user_id: str,
    amount: int,
    description: str,
    video_id: str,
    clip_id: str,
    charge_source: str = "owner_wallet",
    team_id: str | None = None,
    billing_owner_user_id: str | None = None,
    actor_user_id: str | None = None,
    job_id: str | None = None,
    processing_ref: str | None = None,
    usage_metadata: dict[str, Any] | None = None,
):
    amount = int(amount)
    if amount < 0:
        raise RuntimeError(
            f"Failed to charge clip-generation credits: amount must be >= 0 (got {amount})"
        )
    if amount == 0:
        logger.info(
            "Skipping clip-generation charge because resolved amount is 0 "
            "(user=%s clip=%s source=%s)",
            billing_owner_user_id or user_id,
            clip_id,
            charge_source,
        )
        return

    owner_user_id = billing_owner_user_id or user_id
    actor_id = actor_user_id or user_id
    if charge_source == "team_wallet":
        if not team_id:
            raise RuntimeError("Failed to charge clip-generation credits: missing team_id")
        _charge_team_credits_or_raise(
            owner_user_id=owner_user_id,
            team_id=team_id,
            amount=amount,
            owner_tx_type="clip_generation",
            description=description,
            actor_user_id=actor_id,
            video_id=video_id,
            clip_id=clip_id,
            job_id=job_id,
            processing_ref=processing_ref,
            context="Failed to charge clip-generation credits",
        )
        _emit_usage_event_after_charge(
            event_name=POLAR_USAGE_EVENT_GENERATION_NAME,
            category="clip_generation",
            owner_user_id=owner_user_id,
            actor_user_id=actor_id,
            amount=amount,
            charge_source=charge_source,
            team_id=team_id,
            job_id=job_id,
            video_id=video_id,
            clip_id=clip_id,
            usage_metadata=usage_metadata,
        )
        return

    _charge_credits_or_raise(
        user_id=owner_user_id,
        amount=amount,
        tx_type="clip_generation",
        description=description,
        video_id=video_id,
        clip_id=clip_id,
        processing_ref=processing_ref,
        context="Failed to charge clip-generation credits",
    )
    _emit_usage_event_after_charge(
        event_name=POLAR_USAGE_EVENT_GENERATION_NAME,
        category="clip_generation",
        owner_user_id=owner_user_id,
        actor_user_id=actor_id,
        amount=amount,
        charge_source=charge_source,
        team_id=team_id,
        job_id=job_id,
        video_id=video_id,
        clip_id=clip_id,
        usage_metadata=usage_metadata,
    )


def charge_video_analysis_credits(
    *,
    user_id: str,
    amount: int,
    description: str,
    video_id: str,
    charge_source: str = "owner_wallet",
    team_id: str | None = None,
    billing_owner_user_id: str | None = None,
    actor_user_id: str | None = None,
    job_id: str | None = None,
    processing_ref: str | None = None,
    usage_metadata: dict[str, Any] | None = None,
):
    amount = int(amount)
    if amount < 0:
        raise RuntimeError(
            f"Failed to charge video-analysis credits: amount must be >= 0 (got {amount})"
        )
    if amount == 0:
        logger.info(
            "Skipping video-analysis charge because resolved amount is 0 "
            "(user=%s video=%s source=%s)",
            billing_owner_user_id or user_id,
            video_id,
            charge_source,
        )
        return

    owner_user_id = billing_owner_user_id or user_id
    actor_id = actor_user_id or user_id
    if charge_source == "team_wallet":
        if not team_id:
            raise RuntimeError("Failed to charge video-analysis credits: missing team_id")
        _charge_team_credits_or_raise(
            owner_user_id=owner_user_id,
            team_id=team_id,
            amount=amount,
            owner_tx_type="video_analysis",
            description=description,
            actor_user_id=actor_id,
            video_id=video_id,
            clip_id=None,
            job_id=job_id,
            processing_ref=processing_ref,
            context="Failed to charge video-analysis credits",
        )
        _emit_usage_event_after_charge(
            event_name=POLAR_USAGE_EVENT_ANALYSIS_NAME,
            category="video_analysis",
            owner_user_id=owner_user_id,
            actor_user_id=actor_id,
            amount=amount,
            charge_source=charge_source,
            team_id=team_id,
            job_id=job_id,
            video_id=video_id,
            clip_id=None,
            usage_metadata=usage_metadata,
        )
        return

    _charge_credits_or_raise(
        user_id=owner_user_id,
        amount=amount,
        tx_type="video_analysis",
        description=description,
        video_id=video_id,
        clip_id=None,
        processing_ref=processing_ref,
        context="Failed to charge video-analysis credits",
    )
    _emit_usage_event_after_charge(
        event_name=POLAR_USAGE_EVENT_ANALYSIS_NAME,
        category="video_analysis",
        owner_user_id=owner_user_id,
        actor_user_id=actor_id,
        amount=amount,
        charge_source=charge_source,
        team_id=team_id,
        job_id=job_id,
        video_id=video_id,
        clip_id=None,
        usage_metadata=usage_metadata,
    )


def emit_video_analysis_usage_event(
    *,
    user_id: str,
    amount: int,
    video_id: str,
    charge_source: str = "owner_wallet",
    team_id: str | None = None,
    billing_owner_user_id: str | None = None,
    actor_user_id: str | None = None,
    job_id: str | None = None,
    usage_metadata: dict[str, Any] | None = None,
) -> None:
    """Emit video-analysis usage telemetry without charging credits."""
    parsed_amount = int(amount)
    if parsed_amount <= 0:
        return

    owner_user_id = billing_owner_user_id or user_id
    actor_id = actor_user_id or user_id
    _emit_usage_event_after_charge(
        event_name=POLAR_USAGE_EVENT_ANALYSIS_NAME,
        category="video_analysis",
        owner_user_id=owner_user_id,
        actor_user_id=actor_id,
        amount=parsed_amount,
        charge_source=charge_source,
        team_id=team_id,
        job_id=job_id,
        video_id=video_id,
        clip_id=None,
        usage_metadata=usage_metadata,
    )


def update_job_status(
    job_id: str,
    status: str,
    progress: int,
    error: str = None,
    result_data: dict = None,
    action: str | None = None,
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
    if action:
        data["action"] = action

    resp = supabase.table("jobs").update(data).eq("id", job_id).execute()
    if action:
        resp_error = getattr(resp, "error", None)
        if resp_error and _is_missing_column_error(resp_error, "action"):
            fallback_data = dict(data)
            fallback_data.pop("action", None)
            resp = supabase.table("jobs").update(fallback_data).eq("id", job_id).execute()
    assert_response_ok(resp, f"Failed to update job status for {job_id}")

    if status in {"completed", "failed"}:
        try:
            from utils.redis_client import release_job_admission

            release_job_admission(job_id)
        except Exception as exc:
            logger.warning("Failed to release admission token for %s: %s", job_id, exc)


def update_video_status(video_id: str, status: str, **kwargs):
    """Update video record"""
    data = {"status": status}
    data.update(kwargs)
    resp = supabase.table("videos").update(data).eq("id", video_id).execute()
    assert_response_ok(resp, f"Failed to update video status for {video_id}")

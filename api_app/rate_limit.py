"""Shared API rate-limiter configuration."""

from __future__ import annotations

import os
from urllib.parse import quote
import redis
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api_app.state import logger
from config import REDIS_DB, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT

_redis_auth = f":{quote(REDIS_PASSWORD, safe='')}@" if REDIS_PASSWORD else ""
_rate_limit_storage_uri = f"redis://{_redis_auth}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
_rate_limit_log_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
_rate_limits_enabled = os.getenv("DISABLE_RATE_LIMITS", "").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}


def _get_real_client_ip(request: Request) -> str:
    """Extract the real client IP, handling reverse proxies via X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        # First IP in the chain is the original client
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


_FAIL_OPEN_ON_REDIS_DOWN = os.getenv("RATE_LIMIT_FAIL_OPEN", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _build_limiter() -> Limiter:
    try:
        redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            socket_connect_timeout=1,
        ).ping()
        logger.info("Rate limiter initialized with redis backend: %s", _rate_limit_log_uri)
        return Limiter(
            key_func=_get_real_client_ip,
            storage_uri=_rate_limit_storage_uri,
            enabled=_rate_limits_enabled,
        )
    except Exception as exc:
        if _FAIL_OPEN_ON_REDIS_DOWN:
            logger.warning(
                "Rate limiter redis backend unavailable (%s). "
                "Falling back to in-memory limits.",
                exc,
            )
            return Limiter(key_func=_get_real_client_ip, enabled=_rate_limits_enabled)
        # Fail closed: keep the Redis URI so requests are rejected when
        # Redis is unreachable, rather than silently bypassing rate limits.
        logger.error(
            "Rate limiter redis backend unavailable (%s). "
            "Rate limiting will fail closed (reject requests). "
            "Set RATE_LIMIT_FAIL_OPEN=true to fall back to in-memory limits.",
            exc,
        )
        return Limiter(
            key_func=_get_real_client_ip,
            storage_uri=_rate_limit_storage_uri,
            enabled=_rate_limits_enabled,
        )


limiter = _build_limiter()

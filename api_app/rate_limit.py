"""Shared API rate-limiter configuration."""

from __future__ import annotations

import os
from urllib.parse import quote
import redis
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
            key_func=get_remote_address,
            storage_uri=_rate_limit_storage_uri,
            enabled=_rate_limits_enabled,
        )
    except Exception as exc:
        logger.warning(
            "Rate limiter redis backend unavailable (%s). Falling back to in-memory limits.",
            exc,
        )
        return Limiter(key_func=get_remote_address, enabled=_rate_limits_enabled)


limiter = _build_limiter()

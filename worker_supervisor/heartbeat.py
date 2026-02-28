"""Worker heartbeat utilities for detecting hung worker processes."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

HEARTBEAT_KEY_PREFIX = "clipry:worker:heartbeat:"
HEARTBEAT_TTL_SECONDS = 120  # key auto-expires if worker dies without cleanup


def beat(conn, worker_name: str, ttl_seconds: int = HEARTBEAT_TTL_SECONDS) -> None:
    """Called by a worker process to record a heartbeat."""
    key = f"{HEARTBEAT_KEY_PREFIX}{worker_name}"
    conn.setex(key, ttl_seconds, str(time.time()))


def get_heartbeat_age(conn, worker_name: str) -> float | None:
    """Return seconds since last heartbeat, or None if no heartbeat recorded."""
    key = f"{HEARTBEAT_KEY_PREFIX}{worker_name}"
    raw = conn.get(key)
    if raw is None:
        return None
    try:
        return time.time() - float(raw)
    except (ValueError, TypeError):
        return None


def clear_heartbeat(conn, worker_name: str) -> None:
    """Remove the heartbeat key (e.g., on clean shutdown)."""
    key = f"{HEARTBEAT_KEY_PREFIX}{worker_name}"
    conn.delete(key)

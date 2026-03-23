"""Shared HTTP helpers for social publishing: retry with backoff and connection pooling."""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from typing import Generator

import httpx

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _rewind_if_seekable(value: object) -> None:
    seek = getattr(value, "seek", None)
    if not callable(seek):
        return
    try:
        seek(0)
    except Exception:
        return


def _rewind_request_body(kwargs: dict) -> None:
    for key in ("content", "data"):
        _rewind_if_seekable(kwargs.get(key))

    files = kwargs.get("files")
    if isinstance(files, dict):
        file_items = files.values()
    elif isinstance(files, (list, tuple)):
        file_items = []
        for item in files:
            if isinstance(item, tuple) and len(item) >= 2:
                file_items.append(item[1])
    else:
        file_items = []

    for file_item in file_items:
        if isinstance(file_item, tuple):
            if len(file_item) >= 2:
                _rewind_if_seekable(file_item[1])
            continue
        _rewind_if_seekable(file_item)


def resilient_request(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs,
) -> httpx.Response:
    """Execute an HTTP request with retry and exponential backoff on transient failures.

    Retries on httpx.TransportError (connection errors, timeouts) and
    HTTP 429/500/502/503/504 responses. Non-transient 4xx errors are
    returned immediately without retry.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            _rewind_request_body(kwargs)
            response = httpx.request(method, url, **kwargs)
            if response.status_code not in _TRANSIENT_STATUS_CODES or attempt == max_retries:
                return response
            logger.warning(
                "Transient HTTP %d from %s %s, retrying (%d/%d)",
                response.status_code, method, url, attempt + 1, max_retries,
            )
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            logger.warning(
                "Transport error %s for %s %s, retrying (%d/%d)",
                type(exc).__name__, method, url, attempt + 1, max_retries,
            )

        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = delay * 0.25 * random.random()
        time.sleep(delay + jitter)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry exhausted without a response or exception")


def resilient_client_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs,
) -> httpx.Response:
    """Like resilient_request but uses an existing httpx.Client for connection pooling."""
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            _rewind_request_body(kwargs)
            response = client.request(method, url, **kwargs)
            if response.status_code not in _TRANSIENT_STATUS_CODES or attempt == max_retries:
                return response
            logger.warning(
                "Transient HTTP %d from %s %s, retrying (%d/%d)",
                response.status_code, method, url, attempt + 1, max_retries,
            )
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            logger.warning(
                "Transport error %s for %s %s, retrying (%d/%d)",
                type(exc).__name__, method, url, attempt + 1, max_retries,
            )

        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = delay * 0.25 * random.random()
        time.sleep(delay + jitter)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry exhausted without a response or exception")


@contextmanager
def pooled_client(**kwargs) -> Generator[httpx.Client, None, None]:
    """Context manager yielding an httpx.Client with keep-alive connection pooling."""
    client = httpx.Client(
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        **kwargs,
    )
    try:
        yield client
    finally:
        client.close()

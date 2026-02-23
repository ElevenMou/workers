import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    POLAR_ACCESS_TOKEN,
    POLAR_ENV,
    POLAR_ORGANIZATION_ID,
    POLAR_USAGE_EVENT_TIMEOUT_SECONDS,
    POLAR_USAGE_EVENTS_ENABLED,
)

logger = logging.getLogger(__name__)
_WARNED_DISABLED = False


def _polar_api_url() -> str:
    if POLAR_ENV == "production":
        return "https://api.polar.sh"
    return "https://sandbox-api.polar.sh"


def _is_enabled() -> bool:
    global _WARNED_DISABLED

    if not POLAR_USAGE_EVENTS_ENABLED:
        if not _WARNED_DISABLED:
            logger.info("Polar usage events are disabled (POLAR_USAGE_EVENTS_ENABLED=false).")
            _WARNED_DISABLED = True
        return False

    if not POLAR_ACCESS_TOKEN:
        if not _WARNED_DISABLED:
            logger.warning(
                "POLAR_ACCESS_TOKEN is missing; skipping Polar usage event ingestion."
            )
            _WARNED_DISABLED = True
        return False

    return True


def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
            continue

        # Polar metadata values must be scalar primitives (or specific
        # typed objects). Convert unknown values to strings to keep the
        # payload valid and preserve debugging context.
        normalized[key] = str(value)

    return normalized


def emit_polar_usage_event(
    *,
    event_name: str,
    external_customer_id: str,
    external_id: str,
    metadata: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> bool:
    if not event_name or not external_customer_id or not external_id:
        return False

    if not _is_enabled():
        return False

    timestamp = (occurred_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    ).isoformat()
    payload_event: dict[str, Any] = {
        "name": event_name,
        "external_customer_id": external_customer_id,
        "external_id": external_id,
        "timestamp": timestamp,
    }

    if POLAR_ORGANIZATION_ID:
        payload_event["organization_id"] = POLAR_ORGANIZATION_ID

    normalized_metadata = _normalize_metadata(metadata)
    if normalized_metadata:
        payload_event["metadata"] = normalized_metadata

    body = {"events": [payload_event]}
    request_data = json.dumps(body).encode("utf-8")
    request = Request(
        url=f"{_polar_api_url()}/v1/events/ingest",
        data=request_data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {POLAR_ACCESS_TOKEN}",
        },
    )

    timeout_seconds = max(1.0, float(POLAR_USAGE_EVENT_TIMEOUT_SECONDS))

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
            response_text = response_bytes.decode("utf-8") if response_bytes else ""

            if response.status != 200:
                logger.warning(
                    "Polar usage ingest returned unexpected status %s for %s (%s).",
                    response.status,
                    event_name,
                    external_id,
                )
                return False

            if not response_text:
                return True

            try:
                response_payload = json.loads(response_text)
                inserted = int(response_payload.get("inserted") or 0)
                duplicates = int(response_payload.get("duplicates") or 0)
                return inserted > 0 or duplicates > 0
            except Exception:
                # Response shape is non-critical for processing jobs.
                return True

    except HTTPError as exc:
        detail = ""
        try:
            detail_bytes = exc.read()
            detail = detail_bytes.decode("utf-8") if detail_bytes else ""
        except Exception:
            detail = str(exc)

        logger.warning(
            "Failed to ingest Polar usage event %s (%s): HTTP %s %s",
            event_name,
            external_id,
            getattr(exc, "code", "unknown"),
            detail,
        )
    except URLError as exc:
        logger.warning(
            "Failed to ingest Polar usage event %s (%s): %s",
            event_name,
            external_id,
            exc,
        )
    except Exception as exc:
        logger.warning(
            "Unexpected Polar usage ingest error for %s (%s): %s",
            event_name,
            external_id,
            exc,
        )

    return False

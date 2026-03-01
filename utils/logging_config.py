"""Structured JSON logging with correlation ID support."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from typing import Any

correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "job_id", default=None
)


def generate_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    """Outputs each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": record.process,
        }

        cid = correlation_id_var.get(None)
        if cid:
            log_entry["correlation_id"] = cid

        jid = job_id_var.get(None)
        if jid:
            log_entry["job_id"] = jid

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        extra_keys = {"correlation_id", "job_id"}
        for key in extra_keys:
            val = getattr(record, key, None)
            if val and key not in log_entry:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Readable text format with correlation/job IDs when available."""

    def format(self, record: logging.LogRecord) -> str:
        cid = correlation_id_var.get(None) or ""
        jid = job_id_var.get(None) or ""
        ctx_parts = []
        if cid:
            ctx_parts.append(f"cid={cid}")
        if jid:
            ctx_parts.append(f"job={jid}")
        ctx = f" [{' '.join(ctx_parts)}]" if ctx_parts else ""

        ts = self.formatTime(record)
        base = f"{ts}  {record.name:<28s} [{record.process}] {record.levelname:<7s}{ctx} {record.getMessage()}"

        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)

        return base


def setup_logging(*, component: str = "worker") -> None:
    """Configure root logger and Sentry. Call once at process start.

    Uses JSON format when LOG_FORMAT=json (default in Docker),
    human-readable otherwise.
    """
    log_format = os.getenv("LOG_FORMAT", "json").strip().lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
    if sentry_dsn:
        try:
            import sentry_sdk

            integrations = []
            if component == "api":
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                integrations.append(FastApiIntegration())
            else:
                from sentry_sdk.integrations.rq import RqIntegration
                integrations.append(RqIntegration())

            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
                integrations=integrations,
                send_default_pii=False,
            )
            logging.getLogger(__name__).info(
                "Sentry initialized for component=%s", component,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to initialize Sentry: %s", exc,
            )

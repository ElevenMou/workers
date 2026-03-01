"""Sentry context helpers for enriching job error reports.

Call ``configure_job_scope`` at the start of every RQ task to tag
the current Sentry scope with user, job, and resource identifiers.
All subsequent ``capture_exception`` / ``capture_message`` calls
in that task will inherit this context automatically.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _sdk_available() -> bool:
    try:
        import sentry_sdk  # noqa: F401
        return True
    except ImportError:
        return False


def configure_job_scope(
    *,
    job_id: str | None = None,
    job_type: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    clip_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Tag the current Sentry isolation scope with job metadata."""
    if not _sdk_available():
        return

    import sentry_sdk

    scope = sentry_sdk.get_current_scope()

    if user_id:
        scope.set_user({"id": user_id})

    tags: dict[str, str] = {}
    if job_id:
        tags["job_id"] = job_id
    if job_type:
        tags["job_type"] = job_type
    if video_id:
        tags["video_id"] = video_id
    if clip_id:
        tags["clip_id"] = clip_id
    for key, value in tags.items():
        scope.set_tag(key, value)

    context_data: dict[str, Any] = {**tags}
    if extra:
        context_data.update(extra)
    if context_data:
        scope.set_context("job", context_data)


def capture_exception_with_context(
    error: BaseException,
    *,
    job_id: str | None = None,
    job_type: str | None = None,
    user_id: str | None = None,
    video_id: str | None = None,
    clip_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture an exception to Sentry with optional job context override."""
    if not _sdk_available():
        return

    import sentry_sdk

    tags: dict[str, str] = {}
    if job_id:
        tags["job_id"] = job_id
    if job_type:
        tags["job_type"] = job_type
    if video_id:
        tags["video_id"] = video_id
    if clip_id:
        tags["clip_id"] = clip_id

    sentry_sdk.capture_exception(
        error,
        tags=tags or None,
        extras={**(extra or {}), "user_id": user_id} if user_id else extra or None,
    )

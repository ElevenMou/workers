"""Shared dataclasses and errors for social publishing providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class SocialAccountTokens:
    access_token: str
    refresh_token: str | None = None
    token_expires_at: datetime | None = None


@dataclass
class SocialAccountContext:
    id: str
    provider: str
    external_account_id: str
    display_name: str
    handle: str | None
    provider_metadata: dict[str, Any]
    scopes: list[str]
    tokens: SocialAccountTokens


@dataclass
class PublicationContext:
    id: str
    clip_id: str
    clip_title: str | None
    caption: str
    youtube_title: str | None
    scheduled_for: datetime  # Always resolved to UTC by the task layer; never None at runtime


@dataclass
class PublicationMedia:
    local_path: str
    file_size: int
    content_type: str
    signed_url: str | None
    width: int | None
    height: int | None
    duration_seconds: float | None

    @property
    def is_vertical(self) -> bool:
        if self.width is None or self.height is None:
            return False
        return self.height >= self.width


@dataclass
class PublicationResult:
    remote_post_id: str
    remote_post_url: str | None = None
    result_payload: dict[str, Any] | None = None
    provider_payload: dict[str, Any] | None = None
    updated_access_token: str | None = None
    updated_refresh_token: str | None = None
    updated_token_expires_at: datetime | None = None


class SocialProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        recoverable: bool = False,
        refresh_required: bool = False,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.refresh_required = refresh_required
        self.provider_payload = provider_payload or {}

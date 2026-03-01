"""TikTok direct-post publishing integration."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)

_TIKTOK_BASE = "https://open.tiktokapis.com"


def _tiktok_client_key() -> str:
    value = (os.getenv("TIKTOK_CLIENT_KEY") or "").strip()
    if not value:
        raise SocialProviderError("TIKTOK_CLIENT_KEY is not configured.", code="missing_tiktok_client_key")
    return value


def _tiktok_client_secret() -> str:
    value = (os.getenv("TIKTOK_CLIENT_SECRET") or "").strip()
    if not value:
        raise SocialProviderError("TIKTOK_CLIENT_SECRET is not configured.", code="missing_tiktok_client_secret")
    return value


def refresh_access_token(account: SocialAccountContext) -> PublicationResult | None:
    if not account.tokens.refresh_token:
        return None

    response = httpx.post(
        f"{_TIKTOK_BASE}/v2/oauth/token/",
        data={
            "client_key": _tiktok_client_key(),
            "client_secret": _tiktok_client_secret(),
            "grant_type": "refresh_token",
            "refresh_token": account.tokens.refresh_token,
        },
        timeout=30.0,
    )
    payload = response.json()
    if response.status_code >= 400:
        raise SocialProviderError(
            payload.get("error_description") or payload.get("message") or "TikTok token refresh failed.",
            code="tiktok_token_refresh_failed",
            refresh_required=True,
            provider_payload=payload,
        )

    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    refresh_token = payload.get("refresh_token") or account.tokens.refresh_token
    if not access_token:
        raise SocialProviderError(
            "TikTok token refresh did not return an access token.",
            code="tiktok_token_refresh_missing_access_token",
            refresh_required=True,
            provider_payload=payload,
        )

    expiry = None
    try:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    except (TypeError, ValueError):
        expiry = None

    return PublicationResult(
        remote_post_id="",
        updated_access_token=str(access_token),
        updated_refresh_token=str(refresh_token) if refresh_token else None,
        updated_token_expires_at=expiry,
        provider_payload=payload,
    )


def _authorized_post(path: str, *, token: str, json_body: dict) -> dict:
    response = httpx.post(
        f"{_TIKTOK_BASE}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json=json_body,
        timeout=60.0,
    )
    payload = response.json()
    if response.status_code >= 400:
        error = payload.get("error") or {}
        raise SocialProviderError(
            error.get("message") or "TikTok API request failed.",
            code=str(error.get("code") or "tiktok_api_error"),
            refresh_required=response.status_code == 401,
            provider_payload=payload,
        )
    return payload if isinstance(payload, dict) else {"raw": payload}


def _query_creator_info(account: SocialAccountContext) -> dict:
    payload = _authorized_post(
        "/v2/post/publish/creator_info/query/",
        token=account.tokens.access_token,
        json_body={},
    )
    error = payload.get("error") or {}
    if error.get("code") not in {None, "", "ok"}:
        raise SocialProviderError(
            error.get("message") or "TikTok creator info query failed.",
            code=str(error.get("code")),
            provider_payload=payload,
        )
    return payload.get("data") or {}


def _fetch_publish_status(account: SocialAccountContext, publish_id: str) -> dict:
    payload = _authorized_post(
        "/v2/post/publish/status/fetch/",
        token=account.tokens.access_token,
        json_body={"publish_id": publish_id},
    )
    error = payload.get("error") or {}
    if error.get("code") not in {None, "", "ok"}:
        raise SocialProviderError(
            error.get("message") or "TikTok status fetch failed.",
            code=str(error.get("code")),
            provider_payload=payload,
        )
    return payload.get("data") or {}


def publish_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    if not media.signed_url:
        raise SocialProviderError(
            "TikTok direct post requires a signed clip URL.",
            code="tiktok_missing_signed_url",
            recoverable=True,
        )

    creator_info = _query_creator_info(account)
    max_duration = creator_info.get("max_video_post_duration_sec")
    try:
        max_duration_seconds = int(max_duration) if max_duration is not None else None
    except (TypeError, ValueError):
        max_duration_seconds = None
    if (
        max_duration_seconds is not None
        and media.duration_seconds is not None
        and media.duration_seconds > float(max_duration_seconds)
    ):
        raise SocialProviderError(
            f"TikTok account only supports videos up to {max_duration_seconds} seconds.",
            code="tiktok_duration_limit_exceeded",
            provider_payload={"creator_info": creator_info},
        )

    privacy_options = creator_info.get("privacy_level_options") or []
    privacy_level = "SELF_ONLY" if "SELF_ONLY" in privacy_options else (privacy_options[0] if privacy_options else "SELF_ONLY")

    init_payload = _authorized_post(
        "/v2/post/publish/video/init/",
        token=account.tokens.access_token,
        json_body={
            "post_info": {
                "title": (publication.caption or "")[:2200],
                "privacy_level": privacy_level,
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": media.signed_url,
            },
        },
    )

    data = init_payload.get("data") or {}
    publish_id = data.get("publish_id")
    if not publish_id:
        raise SocialProviderError(
            "TikTok direct post initialization did not return a publish id.",
            code="tiktok_missing_publish_id",
            provider_payload=init_payload,
        )

    deadline = time.monotonic() + 180
    last_status_payload: dict = {}
    while time.monotonic() < deadline:
        status_payload = _fetch_publish_status(account, str(publish_id))
        last_status_payload = status_payload
        status = str(status_payload.get("status") or "").upper()
        if status == "PUBLISH_COMPLETE":
            public_ids = status_payload.get("publicaly_available_post_id") or []
            remote_post_id = str(public_ids[0]) if public_ids else str(publish_id)
            remote_post_url = None
            if public_ids and account.handle:
                remote_post_url = f"https://www.tiktok.com/@{account.handle}/video/{public_ids[0]}"
            return PublicationResult(
                remote_post_id=remote_post_id,
                remote_post_url=remote_post_url,
                provider_payload={
                    "creator_info": creator_info,
                    "init": init_payload,
                    "status": status_payload,
                },
                result_payload={"platform": "tiktok", "publish_id": str(publish_id)},
            )
        if status == "SEND_TO_USER_INBOX":
            raise SocialProviderError(
                "TikTok returned inbox delivery instead of direct publish. Reconnect the account or review app approval.",
                code="tiktok_inbox_only",
                provider_payload={
                    "creator_info": creator_info,
                    "init": init_payload,
                    "status": status_payload,
                },
            )
        if status == "FAILED":
            raise SocialProviderError(
                status_payload.get("fail_reason") or "TikTok direct post failed.",
                code="tiktok_publish_failed",
                provider_payload={
                    "creator_info": creator_info,
                    "init": init_payload,
                    "status": status_payload,
                },
            )
        time.sleep(5)

    raise SocialProviderError(
        "Timed out waiting for TikTok to finish direct publishing.",
        code="tiktok_publish_timeout",
        recoverable=True,
        provider_payload={
            "creator_info": creator_info,
            "init": init_payload,
            "status": last_status_payload,
        },
    )

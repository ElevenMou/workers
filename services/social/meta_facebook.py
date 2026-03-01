"""Facebook Page video publishing integration."""

from __future__ import annotations

import os

import httpx

from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)

_META_API_VERSION = (os.getenv("META_GRAPH_API_VERSION") or "v23.0").strip() or "v23.0"
_GRAPH_VIDEO_BASE = f"https://graph-video.facebook.com/{_META_API_VERSION}"
_GRAPH_BASE = f"https://graph.facebook.com/{_META_API_VERSION}"


def _coerce_meta_error(payload: dict, default_message: str) -> SocialProviderError:
    error = payload.get("error") or {}
    message = error.get("message") or default_message
    code = error.get("code")
    refresh_required = bool(code == 190 or error.get("type") == "OAuthException")
    return SocialProviderError(
        message,
        code="meta_page_publish_failed",
        refresh_required=refresh_required,
        provider_payload=payload,
    )


def publish_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    if not media.signed_url:
        raise SocialProviderError(
            "Facebook publishing requires a signed clip URL.",
            code="facebook_missing_signed_url",
            recoverable=True,
        )

    response = httpx.post(
        f"{_GRAPH_VIDEO_BASE}/{account.external_account_id}/videos",
        data={
            "access_token": account.tokens.access_token,
            "file_url": media.signed_url,
            "description": publication.caption or "",
            "title": (publication.caption or "")[:100] or "Clipry clip",
            "published": "true",
        },
        timeout=120.0,
    )
    payload = response.json()
    if response.status_code >= 400:
        raise _coerce_meta_error(payload if isinstance(payload, dict) else {}, "Facebook video publish failed.")

    video_id = payload.get("id")
    permalink = None
    if video_id:
        try:
            permalink_resp = httpx.get(
                f"{_GRAPH_BASE}/{video_id}",
                params={
                    "access_token": account.tokens.access_token,
                    "fields": "permalink_url",
                },
                timeout=30.0,
            )
            permalink_payload = permalink_resp.json()
            if permalink_resp.status_code < 400 and isinstance(permalink_payload, dict):
                permalink = permalink_payload.get("permalink_url")
        except Exception:
            permalink = None

    if not video_id:
        raise SocialProviderError(
            "Facebook video publish succeeded without an id.",
            code="facebook_missing_video_id",
            provider_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )

    return PublicationResult(
        remote_post_id=str(video_id),
        remote_post_url=permalink,
        provider_payload=payload if isinstance(payload, dict) else {"raw": payload},
        result_payload={"platform": "facebook_page", "video_id": str(video_id)},
    )

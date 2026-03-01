"""Instagram Business Reel publishing integration."""

from __future__ import annotations

import os
import time

import httpx

from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)

_META_API_VERSION = (os.getenv("META_GRAPH_API_VERSION") or "v23.0").strip() or "v23.0"
_GRAPH_BASE = f"https://graph.facebook.com/{_META_API_VERSION}"


def _coerce_meta_error(payload: dict, default_message: str, code: str) -> SocialProviderError:
    error = payload.get("error") or {}
    message = error.get("message") or default_message
    error_code = error.get("code")
    refresh_required = bool(error_code == 190 or error.get("type") == "OAuthException")
    return SocialProviderError(
        message,
        code=code,
        refresh_required=refresh_required,
        provider_payload=payload,
    )


def _poll_container_status(*, creation_id: str, access_token: str) -> dict:
    deadline = time.monotonic() + 180
    last_payload: dict = {}
    while time.monotonic() < deadline:
        response = httpx.get(
            f"{_GRAPH_BASE}/{creation_id}",
            params={
                "access_token": access_token,
                "fields": "status_code,status",
            },
            timeout=30.0,
        )
        payload = response.json()
        last_payload = payload if isinstance(payload, dict) else {"raw": payload}
        if response.status_code >= 400:
            raise _coerce_meta_error(last_payload, "Instagram reel status polling failed.", "instagram_status_failed")

        status_code = str(last_payload.get("status_code") or last_payload.get("status") or "").upper()
        if status_code in {"FINISHED", "READY"}:
            return last_payload
        if status_code in {"ERROR", "FAILED", "EXPIRED"}:
            raise SocialProviderError(
                f"Instagram reel container failed with status {status_code}.",
                code="instagram_container_failed",
                provider_payload=last_payload,
            )
        time.sleep(5)

    raise SocialProviderError(
        "Instagram reel container did not become ready before timeout.",
        code="instagram_container_timeout",
        recoverable=True,
        provider_payload=last_payload,
    )


def publish_reel(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    if not media.signed_url:
        raise SocialProviderError(
            "Instagram publishing requires a signed clip URL.",
            code="instagram_missing_signed_url",
            recoverable=True,
        )

    create_response = httpx.post(
        f"{_GRAPH_BASE}/{account.external_account_id}/media",
        data={
            "access_token": account.tokens.access_token,
            "media_type": "REELS",
            "video_url": media.signed_url,
            "caption": publication.caption or "",
        },
        timeout=60.0,
    )
    create_payload = create_response.json()
    if create_response.status_code >= 400:
        raise _coerce_meta_error(
            create_payload if isinstance(create_payload, dict) else {},
            "Instagram reel container creation failed.",
            "instagram_container_create_failed",
        )

    creation_id = create_payload.get("id")
    if not creation_id:
        raise SocialProviderError(
            "Instagram reel container creation did not return an id.",
            code="instagram_container_missing_id",
            provider_payload=create_payload if isinstance(create_payload, dict) else {"raw": create_payload},
        )

    _poll_container_status(
        creation_id=str(creation_id),
        access_token=account.tokens.access_token,
    )

    publish_response = httpx.post(
        f"{_GRAPH_BASE}/{account.external_account_id}/media_publish",
        data={
            "access_token": account.tokens.access_token,
            "creation_id": creation_id,
        },
        timeout=60.0,
    )
    publish_payload = publish_response.json()
    if publish_response.status_code >= 400:
        raise _coerce_meta_error(
            publish_payload if isinstance(publish_payload, dict) else {},
            "Instagram reel publish failed.",
            "instagram_publish_failed",
        )

    media_id = publish_payload.get("id")
    permalink = None
    if media_id:
        try:
            permalink_resp = httpx.get(
                f"{_GRAPH_BASE}/{media_id}",
                params={
                    "access_token": account.tokens.access_token,
                    "fields": "permalink",
                },
                timeout=30.0,
            )
            permalink_payload = permalink_resp.json()
            if permalink_resp.status_code < 400 and isinstance(permalink_payload, dict):
                permalink = permalink_payload.get("permalink")
        except Exception:
            permalink = None

    if not media_id:
        raise SocialProviderError(
            "Instagram reel publish succeeded without a media id.",
            code="instagram_missing_media_id",
            provider_payload=publish_payload if isinstance(publish_payload, dict) else {"raw": publish_payload},
        )

    return PublicationResult(
        remote_post_id=str(media_id),
        remote_post_url=permalink,
        provider_payload={
            "container": create_payload,
            "publish": publish_payload,
        },
        result_payload={"platform": "instagram_business", "media_id": str(media_id)},
    )

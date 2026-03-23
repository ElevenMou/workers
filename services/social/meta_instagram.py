"""Instagram Business Reel publishing integration."""

from __future__ import annotations

import json
import logging
import os
import time
from urllib.parse import urlparse

import httpx

from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)
from services.social.http import pooled_client, resilient_client_request, resilient_request
from services.social.media import is_publicly_reachable_url

logger = logging.getLogger(__name__)

_META_API_VERSION = (os.getenv("META_GRAPH_API_VERSION") or "v25.0").strip() or "v25.0"
_GRAPH_BASE = f"https://graph.facebook.com/{_META_API_VERSION}"


def _read_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    return payload if isinstance(payload, dict) else {"raw": payload}


def _debug_info_message(payload: dict) -> str | None:
    debug_info = payload.get("debug_info")
    if not isinstance(debug_info, dict):
        return None

    raw_message = str(debug_info.get("message") or "").strip()
    if not raw_message:
        return None

    if raw_message.startswith("{") and raw_message.endswith("}"):
        try:
            nested_payload = json.loads(raw_message)
        except ValueError:
            return raw_message
        nested_error = nested_payload.get("error") if isinstance(nested_payload, dict) else None
        nested_message = (
            str(nested_error.get("message") or "").strip()
            if isinstance(nested_error, dict)
            else ""
        )
        if nested_message:
            return nested_message
    return raw_message


def _coerce_meta_error(
    payload: dict,
    default_message: str,
    code: str,
    *,
    recoverable: bool = False,
) -> SocialProviderError:
    error = payload.get("error") or {}
    message = error.get("message") or _debug_info_message(payload) or default_message
    error_code = error.get("code")
    refresh_required = bool(error_code == 190 or error.get("type") == "OAuthException")
    return SocialProviderError(
        message,
        code=code,
        recoverable=recoverable,
        refresh_required=refresh_required,
        provider_payload=payload,
    )


def _instagram_media_url_host(media: PublicationMedia) -> str | None:
    try:
        parsed = urlparse(str(media.signed_url or "").strip())
    except Exception:
        return None
    return parsed.netloc or None


def _container_failure_message(status_payload: dict, status_code: str) -> str:
    details: list[str] = []
    for key in ("error_message", "message", "status"):
        raw_value = status_payload.get(key)
        value = str(raw_value or "").strip()
        if not value or value.upper() == status_code or value in details:
            continue
        details.append(value)

    suffix = f" {' '.join(details)}" if details else ""
    return f"Instagram reel container failed with status {status_code}.{suffix}"


def _create_video_url_reel_container(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    video_url: str,
) -> tuple[str, dict]:
    response = resilient_request(
        "POST",
        f"{_GRAPH_BASE}/{account.external_account_id}/media",
        data={
            "access_token": account.tokens.access_token,
            "media_type": "REELS",
            "video_url": video_url,
            "caption": publication.caption or "",
        },
        timeout=60.0,
    )
    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Instagram reel container creation failed.",
            "instagram_container_create_failed",
            recoverable=True,
        )

    creation_id = str(payload.get("id") or "").strip()
    if not creation_id:
        raise SocialProviderError(
            "Instagram reel container did not return an id.",
            code="instagram_container_missing_id",
            provider_payload=payload,
        )

    return creation_id, payload


def _poll_container_status(*, creation_id: str, access_token: str) -> dict:
    deadline = time.monotonic() + 180
    last_payload: dict = {}
    poll_delay = 2.0
    with pooled_client(timeout=30.0) as client:
        while time.monotonic() < deadline:
            response = resilient_client_request(
                client,
                "GET",
                f"{_GRAPH_BASE}/{creation_id}",
                params={
                    "access_token": access_token,
                    "fields": "status_code,status",
                },
            )
            payload = _read_payload(response)
            last_payload = payload
            if response.status_code >= 400:
                raise _coerce_meta_error(
                    last_payload,
                    "Instagram reel status polling failed.",
                    "instagram_status_failed",
                    recoverable=True,
                )

            status_code = str(last_payload.get("status_code") or last_payload.get("status") or "").upper()
            logger.debug("Instagram container status: %s (creation_id=%s)", status_code, creation_id)
            if status_code in {"FINISHED", "READY"}:
                return last_payload
            if status_code in {"ERROR", "FAILED", "EXPIRED"}:
                raise SocialProviderError(
                    _container_failure_message(last_payload, status_code),
                    code="instagram_container_failed",
                    provider_payload={
                        "creation_id": creation_id,
                        "status": last_payload,
                    },
                )
            time.sleep(poll_delay)
            poll_delay = min(poll_delay + 1.0, 8.0)

    raise SocialProviderError(
        "Instagram reel container did not become ready before timeout.",
        code="instagram_container_timeout",
        recoverable=True,
        provider_payload={
            "creation_id": creation_id,
            "status": last_payload,
        },
    )


def publish_reel(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    public_media_url = str(media.signed_url or "").strip()
    if not is_publicly_reachable_url(public_media_url):
        raise SocialProviderError(
            "Instagram publishing requires a publicly reachable clip URL. Configure a public worker or storage media endpoint before publishing.",
            code="instagram_public_media_url_unavailable",
            recoverable=True,
            provider_payload={
                "media_url_host": _instagram_media_url_host(media),
                "has_local_file": bool(media.local_path),
            },
        )

    logger.info(
        "Starting Instagram reel publish for publication %s (%d bytes, %.1fs)",
        publication.id,
        media.file_size,
        media.duration_seconds or 0,
    )

    if media.duration_seconds is not None and media.duration_seconds > 900.0:
        raise SocialProviderError(
            "Instagram Reels requires clips that are 15 minutes or shorter.",
            code="instagram_reels_duration_exceeded",
        )

    create_payload: dict = {}
    status_payload: dict = {}
    creation_id = ""
    upload_mode = "video_url"
    upload_payload = {"source": "video_url"}

    creation_id, create_payload = _create_video_url_reel_container(
        account=account,
        publication=publication,
        video_url=public_media_url,
    )
    logger.info("Instagram container created: creation_id=%s", creation_id)

    try:
        status_payload = _poll_container_status(
            creation_id=str(creation_id),
            access_token=account.tokens.access_token,
        )
    except SocialProviderError as exc:
        if exc.code in {"instagram_status_failed", "instagram_container_failed", "instagram_container_timeout"}:
            raise SocialProviderError(
                str(exc),
                code=exc.code,
                recoverable=exc.recoverable,
                refresh_required=exc.refresh_required,
                provider_payload={
                    "container": create_payload if isinstance(create_payload, dict) else {"raw": create_payload},
                    "upload": upload_payload,
                    "status": exc.provider_payload,
                    "upload_mode": upload_mode,
                    "media_url_host": _instagram_media_url_host(media),
                },
            ) from exc
        raise

    publish_response = resilient_request(
        "POST",
        f"{_GRAPH_BASE}/{account.external_account_id}/media_publish",
        data={
            "access_token": account.tokens.access_token,
            "creation_id": creation_id,
        },
        timeout=60.0,
    )
    publish_payload = _read_payload(publish_response)
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
            permalink_resp = resilient_request(
                "GET",
                f"{_GRAPH_BASE}/{media_id}",
                params={
                    "access_token": account.tokens.access_token,
                    "fields": "permalink",
                },
                timeout=30.0,
            )
            permalink_payload = _read_payload(permalink_resp)
            if permalink_resp.status_code < 400 and isinstance(permalink_payload, dict):
                permalink = permalink_payload.get("permalink")
        except Exception:
            permalink = None

    if media_id:
        logger.info("Instagram reel published: media_id=%s, permalink=%s", media_id, permalink)

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
            "upload": upload_payload,
            "status": status_payload,
            "publish": publish_payload,
            "upload_mode": upload_mode,
            "media_url_host": _instagram_media_url_host(media),
        },
        result_payload={"platform": "instagram_business", "media_id": str(media_id)},
    )

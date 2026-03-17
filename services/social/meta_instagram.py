"""Instagram Business Reel publishing integration."""

from __future__ import annotations

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

_META_API_VERSION = (os.getenv("META_GRAPH_API_VERSION") or "v23.0").strip() or "v23.0"
_GRAPH_BASE = f"https://graph.facebook.com/{_META_API_VERSION}"


def _read_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    return payload if isinstance(payload, dict) else {"raw": payload}


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


def _instagram_media_url_host(media: PublicationMedia) -> str | None:
    try:
        parsed = urlparse(str(media.signed_url or "").strip())
    except Exception:
        return None
    return parsed.netloc or None


def _instagram_upload_uri_host(upload_uri: str) -> str | None:
    try:
        parsed = urlparse(str(upload_uri or "").strip())
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


def _create_resumable_reel_container(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
) -> tuple[str, str, dict]:
    response = httpx.post(
        f"{_GRAPH_BASE}/{account.external_account_id}/media",
        data={
            "access_token": account.tokens.access_token,
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": publication.caption or "",
        },
        timeout=60.0,
    )
    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Instagram reel upload session creation failed.",
            "instagram_container_create_failed",
        )

    creation_id = str(payload.get("id") or "").strip()
    upload_uri = str(payload.get("uri") or "").strip()
    if not creation_id or not upload_uri:
        raise SocialProviderError(
            "Instagram reel upload session did not return an id and upload URI.",
            code="instagram_container_missing_upload_session",
            provider_payload=payload,
        )

    return creation_id, upload_uri, payload


def _upload_reel_bytes(
    *,
    upload_uri: str,
    media: PublicationMedia,
    access_token: str,
) -> dict:
    try:
        with open(media.local_path, "rb") as handle:
            response = httpx.post(
                upload_uri,
                headers={
                    "Authorization": f"OAuth {access_token}",
                    "offset": "0",
                    "file_size": str(media.file_size),
                    "Content-Type": "application/octet-stream",
                },
                content=handle.read(),
                timeout=300.0,
            )
    except OSError as exc:
        raise SocialProviderError(
            "Instagram reel upload could not read the generated clip file.",
            code="instagram_file_read_failed",
            recoverable=True,
            provider_payload={"path": media.local_path},
        ) from exc

    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Instagram reel binary upload failed.",
            "instagram_upload_failed",
        )
    return payload


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
        payload = _read_payload(response)
        last_payload = payload
        if response.status_code >= 400:
            raise _coerce_meta_error(last_payload, "Instagram reel status polling failed.", "instagram_status_failed")

        status_code = str(last_payload.get("status_code") or last_payload.get("status") or "").upper()
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
        time.sleep(5)

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
    if not media.local_path:
        raise SocialProviderError(
            "Instagram publishing requires a local clip file.",
            code="instagram_missing_local_file",
            recoverable=True,
        )

    if media.duration_seconds is not None and media.duration_seconds > 900.0:
        raise SocialProviderError(
            "Instagram Reels requires clips that are 15 minutes or shorter.",
            code="instagram_reels_duration_exceeded",
        )

    create_payload: dict = {}
    upload_payload: dict = {}
    status_payload: dict = {}
    creation_id = ""
    upload_uri = ""

    creation_id, upload_uri, create_payload = _create_resumable_reel_container(
        account=account,
        publication=publication,
    )

    try:
        upload_payload = _upload_reel_bytes(
            upload_uri=upload_uri,
            media=media,
            access_token=account.tokens.access_token,
        )
        status_payload = _poll_container_status(
            creation_id=str(creation_id),
            access_token=account.tokens.access_token,
        )
    except SocialProviderError as exc:
        if exc.code in {
            "instagram_upload_failed",
            "instagram_status_failed",
            "instagram_container_failed",
            "instagram_container_timeout",
        }:
            raise SocialProviderError(
                str(exc),
                code=exc.code,
                recoverable=exc.recoverable,
                refresh_required=exc.refresh_required,
                provider_payload={
                    "container": create_payload if isinstance(create_payload, dict) else {"raw": create_payload},
                    "upload": upload_payload if isinstance(upload_payload, dict) else {"raw": upload_payload},
                    "status": exc.provider_payload,
                    "upload_uri_host": _instagram_upload_uri_host(upload_uri),
                    "media_url_host": _instagram_media_url_host(media),
                },
            ) from exc
        raise

    publish_response = httpx.post(
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
            permalink_resp = httpx.get(
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
            "upload_uri_host": _instagram_upload_uri_host(upload_uri),
            "media_url_host": _instagram_media_url_host(media),
        },
        result_payload={"platform": "instagram_business", "media_id": str(media_id)},
    )

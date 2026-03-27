"""TikTok direct-post publishing integration."""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)
from services.social.http import pooled_client, resilient_client_request, resilient_request
from services.social.media import is_publicly_reachable_url

_TIKTOK_BASE = "https://open.tiktokapis.com"
_MIN_UPLOAD_CHUNK_BYTES = 5 * 1024 * 1024
_DEFAULT_UPLOAD_CHUNK_BYTES = 10 * 1024 * 1024
_MAX_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_UPLOAD_CHUNKS = 1000


def _read_json_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text}

    return payload if isinstance(payload, dict) else {"raw": payload}


def _read_oauth_error_message(payload: dict, fallback: str) -> str:
    return str(
        payload.get("error_description")
        or payload.get("message")
        or payload.get("error")
        or fallback
    )


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

    logger.info("Refreshing TikTok access token for account %s", account.id)
    response = resilient_request(
        "POST",
        f"{_TIKTOK_BASE}/v2/oauth/token/",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "client_key": _tiktok_client_key(),
            "client_secret": _tiktok_client_secret(),
            "grant_type": "refresh_token",
            "refresh_token": account.tokens.refresh_token,
        },
        timeout=30.0,
    )
    payload = _read_json_payload(response)
    # TikTok OAuth failures can come back as an error body, so inspect the payload too.
    if response.status_code >= 400 or payload.get("error"):
        raise SocialProviderError(
            _read_oauth_error_message(payload, "TikTok token refresh failed."),
            code="tiktok_token_refresh_failed",
            refresh_required=True,
            provider_payload=payload,
        )

    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    refresh_token = payload.get("refresh_token") or account.tokens.refresh_token
    if not access_token:
        raise SocialProviderError(
            _read_oauth_error_message(payload, "TikTok token refresh did not return an access token."),
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


def _authorized_post(
    path: str,
    *,
    token: str,
    json_body: dict,
    client: httpx.Client | None = None,
) -> dict:
    url = f"{_TIKTOK_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    if client is not None:
        response = resilient_client_request(
            client, "POST", url, headers=headers, json=json_body, timeout=60.0,
        )
    else:
        response = resilient_request(
            "POST", url, headers=headers, json=json_body, timeout=60.0,
        )
    payload = _read_json_payload(response)
    if response.status_code >= 400:
        error = payload.get("error") or {}
        error_code = str(error.get("code") or "tiktok_api_error")
        message = error.get("message") or "TikTok API request failed."
        if error_code == "unaudited_client_can_only_post_to_private_accounts":
            message = (
                "TikTok blocked direct publish because this app is unaudited. "
                "Use a private TikTok account for testing or complete TikTok Content Posting API audit."
            )
        raise SocialProviderError(
            message,
            code=error_code,
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


def _select_privacy_level(privacy_options: list[str]) -> str:
    normalized = [str(option).strip().upper() for option in privacy_options if str(option).strip()]
    if not normalized:
        return "SELF_ONLY"

    # Prefer a visible post by default when the account allows it.
    preference_order = [
        "PUBLIC_TO_EVERYONE",
        "MUTUAL_FOLLOW_FRIENDS",
        "FOLLOWER_OF_CREATOR",
        "SELF_ONLY",
    ]
    for candidate in preference_order:
        if candidate in normalized:
            return candidate

    return normalized[0]


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


def _resolve_public_post_ids(status_payload: dict) -> list[str]:
    raw_ids = status_payload.get("publicaly_available_post_id")
    if raw_ids in (None, ""):
        raw_ids = status_payload.get("publicly_available_post_id")

    if isinstance(raw_ids, list):
        return [str(item).strip() for item in raw_ids if str(item).strip()]
    if isinstance(raw_ids, str) and raw_ids.strip():
        return [raw_ids.strip()]
    return []


def _plan_file_upload(file_size: int) -> tuple[int, int]:
    if file_size <= 0:
        raise SocialProviderError(
            "TikTok upload requires a non-empty clip file.",
            code="tiktok_empty_media_file",
            recoverable=True,
        )

    if file_size <= _MAX_UPLOAD_CHUNK_BYTES:
        return file_size, 1

    chunk_size = _DEFAULT_UPLOAD_CHUNK_BYTES
    # TikTok expects total_chunk_count to be video_size / chunk_size rounded down.
    # Any trailing bytes are absorbed into the final chunk, which may be larger
    # than chunk_size (up to 128 MB).
    total_chunk_count = file_size // chunk_size

    if total_chunk_count > _MAX_UPLOAD_CHUNKS:
        chunk_size = max(_MIN_UPLOAD_CHUNK_BYTES, math.ceil(file_size / _MAX_UPLOAD_CHUNKS))
        if chunk_size > _MAX_UPLOAD_CHUNK_BYTES:
            raise SocialProviderError(
                "TikTok file upload exceeds API chunk limits.",
                code="tiktok_file_too_large",
                provider_payload={"file_size": file_size},
            )
        total_chunk_count = file_size // chunk_size

    if total_chunk_count < 1:
        total_chunk_count = 1

    return chunk_size, total_chunk_count


def _upload_file_chunks(
    *,
    upload_url: str,
    media: PublicationMedia,
    chunk_size: int,
    total_chunk_count: int,
    publish_id: str,
) -> None:
    try:
        with open(media.local_path, "rb") as clip_file:
            for chunk_index in range(total_chunk_count):
                start = chunk_index * chunk_size
                end = (
                    start + chunk_size - 1
                    if chunk_index < total_chunk_count - 1
                    else media.file_size - 1
                )
                length = end - start + 1
                payload = clip_file.read(length)

                logger.debug(
                    "TikTok upload chunk %d/%d (%d bytes, range %d-%d)",
                    chunk_index + 1, total_chunk_count, length, start, end,
                )

                if len(payload) != length:
                    raise SocialProviderError(
                        "TikTok upload read fewer bytes than expected.",
                        code="tiktok_upload_file_read_failed",
                        recoverable=True,
                        provider_payload={
                            "publish_id": publish_id,
                            "chunk_index": chunk_index,
                            "expected_length": length,
                            "actual_length": len(payload),
                        },
                    )

                response = resilient_request(
                    "PUT",
                    upload_url,
                    headers={
                        "Content-Type": media.content_type,
                        "Content-Length": str(length),
                        "Content-Range": f"bytes {start}-{end}/{media.file_size}",
                    },
                    content=payload,
                    timeout=180.0,
                    max_retries=2,
                )
                if response.status_code not in {201, 206}:
                    raise SocialProviderError(
                        "TikTok rejected uploaded video data.",
                        code="tiktok_upload_transfer_failed",
                        recoverable=True,
                        provider_payload={
                            "publish_id": publish_id,
                            "chunk_index": chunk_index,
                            "status_code": response.status_code,
                            "response_body": response.text[:1000],
                        },
                    )
    except OSError as exc:
        raise SocialProviderError(
            "TikTok upload could not read the generated clip file.",
            code="tiktok_upload_file_open_failed",
            recoverable=True,
            provider_payload={"path": media.local_path},
        ) from exc


def _init_video_publish(
    *,
    account: SocialAccountContext,
    caption: str,
    privacy_level: str,
    media: PublicationMedia,
    chunk_size: int | None = None,
    total_chunk_count: int | None = None,
    disable_duet: bool,
    disable_comment: bool,
    disable_stitch: bool,
    brand_content_toggle: bool,
    brand_organic_toggle: bool,
    video_cover_timestamp_ms: int | None = None,
    video_url: str | None = None,
) -> dict:
    source_info = (
        {
            "source": "PULL_FROM_URL",
            "video_url": video_url,
        }
        if video_url
        else {
            "source": "FILE_UPLOAD",
            "video_size": media.file_size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunk_count,
        }
    )
    post_info = {
        "title": caption[:2200],
        "privacy_level": privacy_level,
        "disable_duet": disable_duet,
        "disable_comment": disable_comment,
        "disable_stitch": disable_stitch,
        "brand_content_toggle": brand_content_toggle,
        "brand_organic_toggle": brand_organic_toggle,
    }
    if video_cover_timestamp_ms is not None:
        post_info["video_cover_timestamp_ms"] = video_cover_timestamp_ms

    return _authorized_post(
        "/v2/post/publish/video/init/",
        token=account.tokens.access_token,
        json_body={
            "post_info": post_info,
            "source_info": source_info,
        },
    )


def publish_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    if not media.local_path:
        raise SocialProviderError(
            "TikTok direct post requires a local clip file.",
            code="tiktok_missing_local_file",
            recoverable=True,
        )

    logger.info(
        "Starting TikTok publish for publication %s (%d bytes, %.1fs)",
        publication.id, media.file_size, media.duration_seconds or 0,
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
    tiktok_config = (
        publication.resolved_config
        if publication.resolved_config is not None
        and getattr(publication.resolved_config, "provider", None) == "tiktok"
        else None
    )
    tiktok_settings = getattr(tiktok_config, "tiktok", None)
    privacy_level = (
        str(tiktok_settings.privacyLevel)
        if tiktok_settings is not None
        else _select_privacy_level(list(privacy_options))
    )
    public_media_url = str(media.signed_url or "").strip()
    use_pull_from_url = is_publicly_reachable_url(public_media_url)
    chunk_size, total_chunk_count = (
        _plan_file_upload(media.file_size) if not use_pull_from_url else (None, None)
    )

    caption = publication.caption or ""
    init_payload = _init_video_publish(
        account=account,
        caption=caption,
        privacy_level=privacy_level,
        media=media,
        chunk_size=chunk_size,
        total_chunk_count=total_chunk_count,
        disable_duet=(
            not bool(tiktok_settings.allowDuet)
            if tiktok_settings is not None
            else False
        ),
        disable_comment=(
            not bool(tiktok_settings.allowComment)
            if tiktok_settings is not None
            else False
        ),
        disable_stitch=(
            not bool(tiktok_settings.allowStitch)
            if tiktok_settings is not None
            else False
        ),
        brand_content_toggle=(
            bool(tiktok_settings.brandContentToggle)
            if tiktok_settings is not None
            else False
        ),
        brand_organic_toggle=(
            bool(tiktok_settings.brandOrganicToggle)
            if tiktok_settings is not None
            else False
        ),
        video_cover_timestamp_ms=(
            publication.resolved_config.content.coverTimestampMs
            if tiktok_config is not None
            else None
        ),
        video_url=public_media_url if use_pull_from_url else None,
    )

    data = init_payload.get("data") or {}
    publish_id = data.get("publish_id")
    if not publish_id:
        raise SocialProviderError(
            "TikTok direct post initialization did not return a publish id.",
            code="tiktok_missing_publish_id",
            provider_payload=init_payload,
        )
    logger.info(
        "TikTok publish initialized: publish_id=%s, chunks=%d, chunk_size=%d",
        publish_id, total_chunk_count, chunk_size,
    )
    upload_url = str(data.get("upload_url") or "").strip()
    if not use_pull_from_url:
        if not upload_url:
            raise SocialProviderError(
                "TikTok direct post initialization did not return an upload URL.",
                code="tiktok_missing_upload_url",
                provider_payload=init_payload,
            )

        _upload_file_chunks(
            upload_url=upload_url,
            media=media,
            chunk_size=int(chunk_size or 0),
            total_chunk_count=int(total_chunk_count or 0),
            publish_id=str(publish_id),
        )

    deadline = time.monotonic() + 180
    last_status_payload: dict = {}
    poll_delay = 2.0
    with pooled_client(timeout=60.0) as client:
        while time.monotonic() < deadline:
            status_payload = _authorized_post(
                "/v2/post/publish/status/fetch/",
                token=account.tokens.access_token,
                json_body={"publish_id": str(publish_id)},
                client=client,
            )
            error = (status_payload.get("error") or {})
            if error.get("code") not in {None, "", "ok"}:
                raise SocialProviderError(
                    error.get("message") or "TikTok status fetch failed.",
                    code=str(error.get("code")),
                    provider_payload=status_payload,
                )
            status_data = status_payload.get("data") or {}
            last_status_payload = status_data
            status = str(status_data.get("status") or "").upper()
            logger.debug("TikTok publish status: %s (publish_id=%s)", status, publish_id)
            if status == "PUBLISH_COMPLETE":
                public_ids = _resolve_public_post_ids(status_data)
                remote_post_id = str(public_ids[0]) if public_ids else str(publish_id)
                remote_post_url = None
                if public_ids and account.handle:
                    remote_post_url = f"https://www.tiktok.com/@{account.handle}/video/{public_ids[0]}"
                logger.info("TikTok publish complete: post_id=%s", remote_post_id)
                return PublicationResult(
                    remote_post_id=remote_post_id,
                    remote_post_url=remote_post_url,
                    provider_payload={
                        "creator_info": creator_info,
                        "init": init_payload,
                        "status": status_data,
                        "source": "PULL_FROM_URL" if use_pull_from_url else "FILE_UPLOAD",
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
                        "status": status_data,
                    },
                )
            if status == "FAILED":
                raise SocialProviderError(
                    status_data.get("fail_reason") or "TikTok direct post failed.",
                    code="tiktok_publish_failed",
                    provider_payload={
                        "creator_info": creator_info,
                        "init": init_payload,
                        "status": status_data,
                    },
                )
            time.sleep(poll_delay)
            poll_delay = min(poll_delay + 1.0, 8.0)

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

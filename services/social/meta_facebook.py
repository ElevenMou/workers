"""Facebook publishing integration for connected Pages."""

from __future__ import annotations

import json
import logging
import os
import time

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

_META_API_VERSION = (os.getenv("META_GRAPH_API_VERSION") or "v25.0").strip() or "v25.0"
_GRAPH_BASE = f"https://graph.facebook.com/{_META_API_VERSION}"
_REELS_POLL_INTERVAL_SECONDS = 5
_REELS_PUBLISH_TIMEOUT_SECONDS = 300
_FACEBOOK_REELS_MAX_DURATION_SECONDS = 90.0
_FACEBOOK_REELS_ASPECT_RATIO_TOLERANCE = 0.02


def _read_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    return payload if isinstance(payload, dict) else {"raw": payload}


def _coerce_meta_error(
    payload: dict,
    default_message: str,
    *,
    code: str,
    recoverable: bool = False,
) -> SocialProviderError:
    error = payload.get("error") or {}
    message = error.get("message") or default_message
    error_code = error.get("code")
    refresh_required = bool(error_code == 190 or error.get("type") == "OAuthException")
    return SocialProviderError(
        message,
        code=code,
        refresh_required=refresh_required,
        recoverable=recoverable,
        provider_payload=payload,
    )


def _normalize_status(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _phase_status(phase: object) -> str:
    if isinstance(phase, dict):
        return _normalize_status(phase.get("status") or phase.get("state"))
    if isinstance(phase, str):
        return _normalize_status(phase)
    return ""


def _extract_status(payload: dict) -> dict:
    status = payload.get("status")
    if isinstance(status, dict):
        return status
    return {}


def _upload_processing_failed(status_payload: dict) -> bool:
    status = _extract_status(status_payload)
    return any(
        _phase_status(status.get(key)) in {"error", "failed", "expired"}
        for key in (
            "uploading_phase",
            "processing_phase",
            "publishing_phase",
            "copyright_check_status",
        )
    )


def _uploading_complete(status_payload: dict) -> bool:
    status = _extract_status(status_payload)
    upload_phase = _phase_status(status.get("uploading_phase"))
    if upload_phase in {"complete", "completed", "finished", "success"}:
        return True

    video_status = _normalize_status(status.get("video_status"))
    return video_status in {"uploaded", "ready", "published"}


def _publishing_complete(status_payload: dict) -> bool:
    status = _extract_status(status_payload)
    publishing_phase = _phase_status(status.get("publishing_phase"))
    processing_phase = _phase_status(status.get("processing_phase"))
    video_status = _normalize_status(status.get("video_status"))

    if publishing_phase in {"complete", "completed", "finished", "success"}:
        return True
    if video_status in {"published", "ready"} and processing_phase in {
        "",
        "complete",
        "completed",
        "finished",
        "success",
    }:
        return True

    return False


def _is_near_9_16(width: int, height: int) -> bool:
    """Return True if the aspect ratio is within tolerance of 9:16."""
    target_ratio = 9.0 / 16.0
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) <= _FACEBOOK_REELS_ASPECT_RATIO_TOLERANCE


def _is_reel_eligible(media: PublicationMedia) -> bool:
    """Return True if the clip meets Facebook Reel requirements (<=90s, ~9:16)."""
    if media.duration_seconds is None or media.duration_seconds > _FACEBOOK_REELS_MAX_DURATION_SECONDS:
        return False
    if media.width is None or media.height is None:
        return False
    if not _is_near_9_16(media.width, media.height):
        return False
    return True


def _validate_reel_media(media: PublicationMedia) -> None:
    if media.duration_seconds is None:
        raise SocialProviderError(
            "Facebook Reels publishing requires a clip duration.",
            code="facebook_reels_missing_duration",
        )
    if media.duration_seconds > _FACEBOOK_REELS_MAX_DURATION_SECONDS:
        raise SocialProviderError(
            f"Facebook Reels requires clips that are {int(_FACEBOOK_REELS_MAX_DURATION_SECONDS)} seconds or shorter.",
            code="facebook_reels_duration_exceeded",
        )
    if media.width is None or media.height is None:
        raise SocialProviderError(
            "Facebook Reels publishing requires clip dimensions.",
            code="facebook_reels_missing_dimensions",
        )
    if not _is_near_9_16(media.width, media.height):
        raise SocialProviderError(
            "Facebook Reels requires a near 9:16 vertical video.",
            code="facebook_reels_invalid_aspect_ratio",
            provider_payload={
                "width": media.width,
                "height": media.height,
            },
        )


def _publication_title(publication: PublicationContext) -> str:
    resolved_title = None
    if publication.resolved_config is not None:
        resolved_title = publication.resolved_config.content.title
    return (resolved_title or publication.clip_title or publication.caption or "").strip()[:100] or "Clipry clip"


def _facebook_config(publication: PublicationContext):
    if (
        publication.resolved_config is None
        or getattr(publication.resolved_config, "provider", None) != "facebook_page"
    ):
        return None
    return publication.resolved_config.facebook


def _facebook_follow_up_requests(publication: PublicationContext) -> dict[str, object]:
    facebook_settings = _facebook_config(publication)
    if facebook_settings is None:
        return {}
    return {
        "collaboratorPageId": (
            str(facebook_settings.collaboratorPageId)
            if facebook_settings.collaboratorPageId
            else None
        ),
        "crosspostPageIds": [str(page_id) for page_id in facebook_settings.crosspostPageIds],
        "allowBusinessManagerCrossposting": bool(
            facebook_settings.allowBusinessManagerCrossposting
        ),
    }


def _start_reel_upload(account: SocialAccountContext) -> tuple[str, str, dict]:
    response = resilient_request(
        "POST",
        f"{_GRAPH_BASE}/me/video_reels",
        params={
            "access_token": account.tokens.access_token,
            "upload_phase": "start",
        },
        timeout=60.0,
    )
    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Facebook Reels upload session creation failed.",
            code="facebook_reels_start_failed",
        )

    video_id = str(payload.get("video_id") or payload.get("id") or "").strip()
    upload_url = str(payload.get("upload_url") or "").strip()
    if not video_id or not upload_url:
        raise SocialProviderError(
            "Facebook Reels upload session did not return a video id and upload URL.",
            code="facebook_reels_start_missing_fields",
            provider_payload=payload,
        )

    return video_id, upload_url, payload


def _upload_reel_bytes(upload_url: str, media: PublicationMedia, access_token: str) -> dict:
    try:
        with open(media.local_path, "rb") as handle:
            response = resilient_request(
                "POST",
                upload_url,
                headers={
                    "Authorization": f"OAuth {access_token}",
                    "offset": "0",
                    "file_size": str(media.file_size),
                    "Content-Type": "application/octet-stream",
                },
                content=handle,
                timeout=300.0,
                max_retries=2,
            )
    except OSError as exc:
        raise SocialProviderError(
            "Facebook Reels upload could not read the generated clip file.",
            code="facebook_reels_file_read_failed",
            recoverable=True,
            provider_payload={"path": media.local_path},
        ) from exc

    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Facebook Reels upload failed.",
            code="facebook_reels_upload_failed",
            recoverable=True,
        )
    return payload


def _fetch_reel_status(
    video_id: str,
    access_token: str,
    *,
    client: httpx.Client | None = None,
) -> dict:
    url = f"{_GRAPH_BASE}/{video_id}"
    params = {"access_token": access_token, "fields": "status"}
    if client is not None:
        response = resilient_client_request(client, "GET", url, params=params)
    else:
        response = resilient_request("GET", url, params=params, timeout=30.0)
    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Facebook Reels status polling failed.",
            code="facebook_reels_status_failed",
            recoverable=True,
        )
    return payload


def _wait_for_upload(video_id: str, access_token: str) -> dict:
    deadline = time.monotonic() + 60
    last_payload: dict = {}
    poll_delay = 2.0
    with pooled_client(timeout=30.0) as client:
        while time.monotonic() < deadline:
            payload = _fetch_reel_status(video_id, access_token, client=client)
            last_payload = payload
            logger.debug("Facebook Reels upload status: %s (video_id=%s)", _extract_status(payload), video_id)
            if _upload_processing_failed(payload):
                raise SocialProviderError(
                    "Facebook Reels upload failed during processing.",
                    code="facebook_reels_upload_processing_failed",
                    provider_payload=payload,
                )
            if _uploading_complete(payload):
                return payload
            time.sleep(poll_delay)
            poll_delay = min(poll_delay + 1.0, 8.0)

    raise SocialProviderError(
        "Facebook Reels upload did not complete before timeout.",
        code="facebook_reels_upload_timeout",
        recoverable=True,
        provider_payload=last_payload,
    )


def _finish_reel_publish(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    video_id: str,
) -> dict:
    title = _publication_title(publication)
    facebook_settings = _facebook_config(publication)
    data = {
        "access_token": account.tokens.access_token,
        "video_id": video_id,
        "upload_phase": "finish",
        "video_state": "PUBLISHED",
        "description": publication.caption or "",
        "title": title,
    }
    if facebook_settings is not None and facebook_settings.placeId:
        data["place"] = str(facebook_settings.placeId)
    response = resilient_request(
        "POST",
        f"{_GRAPH_BASE}/me/video_reels",
        data=data,
        timeout=60.0,
    )
    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Facebook Reels publish failed.",
            code="facebook_reels_finish_failed",
        )
    return payload


def _wait_for_publish(video_id: str, access_token: str) -> dict:
    deadline = time.monotonic() + _REELS_PUBLISH_TIMEOUT_SECONDS
    last_payload: dict = {}
    poll_delay = 2.0
    with pooled_client(timeout=30.0) as client:
        while time.monotonic() < deadline:
            payload = _fetch_reel_status(video_id, access_token, client=client)
            last_payload = payload
            logger.debug("Facebook Reels publish status: %s (video_id=%s)", _extract_status(payload), video_id)
            if _upload_processing_failed(payload):
                raise SocialProviderError(
                    "Facebook Reels publish failed during processing.",
                    code="facebook_reels_processing_failed",
                    provider_payload=payload,
                )
            if _publishing_complete(payload):
                return payload
            time.sleep(poll_delay)
            poll_delay = min(poll_delay + 1.0, 8.0)

    raise SocialProviderError(
        "Facebook Reels did not finish publishing before timeout.",
        code="facebook_reels_publish_timeout",
        recoverable=True,
        provider_payload=last_payload,
    )


def _fetch_reel_permalink(video_id: str, access_token: str) -> str | None:
    try:
        response = resilient_request(
            "GET",
            f"{_GRAPH_BASE}/{video_id}",
            params={
                "access_token": access_token,
                "fields": "permalink_url",
            },
            timeout=30.0,
        )
        payload = _read_payload(response)
        if response.status_code < 400:
            permalink = payload.get("permalink_url")
            if isinstance(permalink, str) and permalink.strip():
                return permalink.strip()
    except Exception:
        return f"https://www.facebook.com/reel/{video_id}"

    return f"https://www.facebook.com/reel/{video_id}"


def _upload_page_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    """Upload as a regular Facebook Page Video (not a Reel)."""
    if media.duration_seconds is None:
        raise SocialProviderError(
            "Facebook publishing requires a clip duration.",
            code="facebook_video_missing_duration",
        )

    page_id = account.external_account_id
    title = _publication_title(publication)
    facebook_settings = _facebook_config(publication)
    data = {
        "access_token": account.tokens.access_token,
        "description": publication.caption or "",
        "title": title,
    }
    if facebook_settings is not None:
        if facebook_settings.placeId:
            data["place"] = str(facebook_settings.placeId)
        if facebook_settings.contentCategory is not None:
            data["content_category"] = str(facebook_settings.contentCategory)
        if facebook_settings.contentTags:
            data["content_tags"] = json.dumps(
                [str(tag) for tag in facebook_settings.contentTags]
            )
        if facebook_settings.hideFromNewsfeed:
            data["hide_from_newsfeed"] = "true"
        if facebook_settings.feedTargeting is not None:
            data["feed_targeting"] = json.dumps(
                facebook_settings.feedTargeting.model_dump(mode="json")
            )
        if facebook_settings.targeting is not None:
            data["targeting"] = json.dumps(
                facebook_settings.targeting.model_dump(mode="json")
            )

    try:
        with open(media.local_path, "rb") as handle:
            response = resilient_request(
                "POST",
                f"{_GRAPH_BASE}/{page_id}/videos",
                data=data,
                files={"source": (os.path.basename(media.local_path), handle, media.content_type)},
                timeout=300.0,
                max_retries=2,
            )
    except OSError as exc:
        raise SocialProviderError(
            "Facebook video upload could not read the generated clip file.",
            code="facebook_video_file_read_failed",
            recoverable=True,
            provider_payload={"path": media.local_path},
        ) from exc

    payload = _read_payload(response)
    if response.status_code >= 400:
        raise _coerce_meta_error(
            payload,
            "Facebook Page Video upload failed.",
            code="facebook_video_upload_failed",
            recoverable=True,
        )

    video_id = str(payload.get("id") or "").strip()
    if not video_id:
        raise SocialProviderError(
            "Facebook Page Video upload did not return a video id.",
            code="facebook_video_missing_id",
            provider_payload=payload,
        )

    permalink = f"https://www.facebook.com/watch/?v={video_id}"
    logger.info("Facebook Page Video upload complete: video_id=%s", video_id)

    return PublicationResult(
        remote_post_id=video_id,
        remote_post_url=permalink,
        provider_payload={
            "upload": payload,
            "requested_follow_ups": _facebook_follow_up_requests(publication),
        },
        result_payload={
            "platform": "facebook_page",
            "video_id": video_id,
            "publish_target": "facebook_page_video",
        },
    )


def publish_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    facebook_settings = _facebook_config(publication)
    publish_target = (
        str(facebook_settings.publishTarget)
        if facebook_settings is not None
        else "auto"
    )

    if publish_target == "reel":
        logger.info(
            "Starting Facebook Reels publish for publication %s (%d bytes, %.1fs)",
            publication.id,
            media.file_size,
            media.duration_seconds or 0,
        )
        _validate_reel_media(media)

        video_id, start_payload, upload_payload = "", {}, {}
        upload_status_payload, finish_payload, publish_status_payload = {}, {}, {}

        video_id, _upload_url, start_payload = _start_reel_upload(account)
        logger.info("Facebook Reels upload session started: video_id=%s", video_id)
        upload_payload = _upload_reel_bytes(_upload_url, media, account.tokens.access_token)
        upload_status_payload = _wait_for_upload(video_id, account.tokens.access_token)
        finish_payload = _finish_reel_publish(
            account=account,
            publication=publication,
            video_id=video_id,
        )
        publish_status_payload = _wait_for_publish(video_id, account.tokens.access_token)
        permalink = _fetch_reel_permalink(video_id, account.tokens.access_token)
        logger.info("Facebook Reels publish complete: video_id=%s, permalink=%s", video_id, permalink)

        return PublicationResult(
            remote_post_id=video_id,
            remote_post_url=permalink,
            provider_payload={
                "start": start_payload,
                "upload": upload_payload,
                "upload_status": upload_status_payload,
                "finish": finish_payload,
                "publish_status": publish_status_payload,
                "requested_follow_ups": _facebook_follow_up_requests(publication),
            },
            result_payload={
                "platform": "facebook_page",
                "video_id": video_id,
                "publish_target": "facebook_reels",
            },
        )

    if publish_target == "page_video":
        logger.info(
            "Starting Facebook Page Video upload for publication %s (%d bytes, %.1fs)",
            publication.id,
            media.file_size,
            media.duration_seconds or 0,
        )
        return _upload_page_video(
            account=account,
            publication=publication,
            media=media,
        )

    if _is_reel_eligible(media):
        logger.info(
            "Starting Facebook Reels publish for publication %s (%d bytes, %.1fs)",
            publication.id, media.file_size, media.duration_seconds or 0,
        )
        _validate_reel_media(media)

        video_id, start_payload, upload_payload = "", {}, {}
        upload_status_payload, finish_payload, publish_status_payload = {}, {}, {}

        video_id, _upload_url, start_payload = _start_reel_upload(account)
        logger.info("Facebook Reels upload session started: video_id=%s", video_id)
        upload_payload = _upload_reel_bytes(_upload_url, media, account.tokens.access_token)
        upload_status_payload = _wait_for_upload(video_id, account.tokens.access_token)
        finish_payload = _finish_reel_publish(
            account=account,
            publication=publication,
            video_id=video_id,
        )
        publish_status_payload = _wait_for_publish(video_id, account.tokens.access_token)
        permalink = _fetch_reel_permalink(video_id, account.tokens.access_token)
        logger.info("Facebook Reels publish complete: video_id=%s, permalink=%s", video_id, permalink)

        return PublicationResult(
            remote_post_id=video_id,
            remote_post_url=permalink,
            provider_payload={
                "start": start_payload,
                "upload": upload_payload,
                "upload_status": upload_status_payload,
                "finish": finish_payload,
                "publish_status": publish_status_payload,
                "requested_follow_ups": _facebook_follow_up_requests(publication),
            },
            result_payload={
                "platform": "facebook_page",
                "video_id": video_id,
                "publish_target": "facebook_reels",
            },
        )

    logger.info(
        "Starting Facebook Page Video upload for publication %s (%d bytes, %.1fs)",
        publication.id, media.file_size, media.duration_seconds or 0,
    )
    return _upload_page_video(
        account=account,
        publication=publication,
        media=media,
    )

"""YouTube Shorts publishing integration.

Uses the YouTube Data API v3 resumable upload protocol:
https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from config import YOUTUBE_SHORTS_MAX_DURATION_SECONDS
from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
    SocialProviderError,
)
from services.social.http import resilient_request

logger = logging.getLogger(__name__)

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

_UPLOAD_INIT_TIMEOUT = 30.0
_UPLOAD_TRANSFER_TIMEOUT = 600.0

_YOUTUBE_ERROR_MAP: dict[str, str] = {
    "quotaExceeded": "YouTube API quota exceeded. Please try again later.",
    "uploadLimitExceeded": "YouTube daily upload limit reached. Please try again tomorrow.",
    "youtubeSignupRequired": "The YouTube channel needs to complete setup at youtube.com before uploading.",
    "forbidden": "Access denied. The YouTube account may need to be reconnected.",
    "invalidTitle": "The video title is invalid. Please use a different title.",
    "invalidDescription": "The video description is invalid. Please shorten or revise it.",
    "invalidMetadata": "The video metadata was rejected by YouTube. Please check the title and description.",
    "videoNotFound": "YouTube could not process the uploaded video.",
    "processingFailure": "YouTube failed to process the video. Please try again.",
    "notFound": "The YouTube resource was not found. The channel or video may have been deleted.",
    "badRequest": "YouTube rejected the request. Please check the video format and metadata.",
    "rateLimitExceeded": "YouTube rate limit reached. Please wait a few minutes and try again.",
    "videoTooLong": "The video exceeds the maximum duration allowed by YouTube.",
    "invalidVideoId": "YouTube returned an invalid video reference. Please try uploading again.",
    "duplicate": "This video has already been uploaded to YouTube.",
}


def _google_client_id() -> str:
    value = (os.getenv("GOOGLE_CLIENT_ID") or "").strip()
    if not value:
        raise SocialProviderError("GOOGLE_CLIENT_ID is not configured.", code="missing_google_client_id")
    return value


def _google_client_secret() -> str:
    value = (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()
    if not value:
        raise SocialProviderError("GOOGLE_CLIENT_SECRET is not configured.", code="missing_google_client_secret")
    return value


def refresh_access_token(account: SocialAccountContext) -> PublicationResult | None:
    if not account.tokens.refresh_token:
        return None

    response = resilient_request(
        "POST",
        _GOOGLE_TOKEN_URL,
        data={
            "client_id": _google_client_id(),
            "client_secret": _google_client_secret(),
            "grant_type": "refresh_token",
            "refresh_token": account.tokens.refresh_token,
        },
        timeout=30.0,
    )
    payload = response.json()
    if response.status_code >= 400:
        error_detail = payload.get("error_description") or payload.get("error") or ""
        if "invalid_grant" in str(error_detail).lower():
            raise SocialProviderError(
                "YouTube authorization has expired. Please reconnect your account.",
                code="google_token_invalid_grant",
                refresh_required=True,
                provider_payload=payload,
            )
        raise SocialProviderError(
            error_detail or "Failed to refresh YouTube access token.",
            code="google_token_refresh_failed",
            refresh_required=True,
            provider_payload=payload,
        )

    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not access_token:
        raise SocialProviderError(
            "Google token refresh did not return an access token.",
            code="google_token_refresh_missing_access_token",
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
        updated_refresh_token=account.tokens.refresh_token,
        updated_token_expires_at=expiry,
        provider_payload=payload,
    )


def _map_youtube_error(status_code: int, payload: dict) -> SocialProviderError:
    """Extract a user-friendly error from the YouTube API response."""
    error_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
    api_message = error_obj.get("message") if isinstance(error_obj, dict) else None
    errors_list = error_obj.get("errors", []) if isinstance(error_obj, dict) else []

    reason = ""
    if errors_list and isinstance(errors_list[0], dict):
        reason = str(errors_list[0].get("reason") or "")

    friendly = _YOUTUBE_ERROR_MAP.get(reason, "")
    message = friendly or api_message or "YouTube upload failed."

    needs_reconnect = status_code in {401, 403} or reason in {"forbidden", "insufficientPermissions"}

    return SocialProviderError(
        message,
        code=f"youtube_{reason}" if reason else "youtube_upload_failed",
        refresh_required=needs_reconnect,
        recoverable=reason in {"quotaExceeded", "uploadLimitExceeded", "processingFailure", "rateLimitExceeded"},
        provider_payload=payload if isinstance(payload, dict) else {"raw": payload},
    )


def _build_metadata(
    title: str,
    caption: str,
    *,
    privacy_status: str,
    self_declared_made_for_kids: bool,
) -> dict:
    description = caption
    if "#Shorts" not in description:
        description = f"{description}\n\n#Shorts".strip()

    return {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": self_declared_made_for_kids,
        },
    }


def _initiate_resumable_upload(
    access_token: str,
    metadata: dict,
    file_size: int,
    content_type: str,
    *,
    notify_subscribers: bool,
) -> str:
    """POST metadata to YouTube and return the resumable upload URI."""
    response = resilient_request(
        "POST",
        _YOUTUBE_UPLOAD_URL,
        params={
            "uploadType": "resumable",
            "part": "snippet,status",
            "notifySubscribers": str(notify_subscribers).lower(),
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": content_type,
        },
        content=json.dumps(metadata),
        timeout=_UPLOAD_INIT_TIMEOUT,
    )

    if response.status_code >= 400:
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}
        raise _map_youtube_error(response.status_code, payload)

    upload_url = response.headers.get("Location")
    if not upload_url:
        raise SocialProviderError(
            "YouTube did not return a resumable upload URL.",
            code="youtube_missing_upload_url",
            provider_payload={"status": response.status_code, "headers": dict(response.headers)},
        )
    return upload_url


def _upload_video_bytes(
    upload_url: str,
    local_path: str,
    file_size: int,
    content_type: str,
) -> dict:
    """PUT the video file to the resumable upload URI and return the response payload."""
    with open(local_path, "rb") as handle:
        response = resilient_request(
            "PUT",
            upload_url,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_size),
            },
            content=handle,
            timeout=_UPLOAD_TRANSFER_TIMEOUT,
            max_retries=2,
        )

    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}

    if response.status_code >= 400:
        raise _map_youtube_error(response.status_code, payload)

    return payload


def publish_video(
    *,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    if not media.is_vertical:
        raise SocialProviderError(
            "YouTube Shorts requires a vertical video.",
            code="youtube_shorts_not_vertical",
        )

    if media.duration_seconds and media.duration_seconds > float(YOUTUBE_SHORTS_MAX_DURATION_SECONDS):
        raise SocialProviderError(
            f"YouTube Shorts must be {YOUTUBE_SHORTS_MAX_DURATION_SECONDS} seconds or shorter.",
            code="youtube_shorts_duration_exceeded",
        )

    title = (publication.youtube_title or publication.caption or "").strip()
    if not title:
        raise SocialProviderError(
            "YouTube publishing requires a title.",
            code="youtube_title_required",
        )

    caption = (publication.caption or "")[:5000]
    youtube_config = (
        publication.resolved_config
        if publication.resolved_config is not None
        and getattr(publication.resolved_config, "provider", None) == "youtube_channel"
        else None
    )
    youtube_settings = getattr(youtube_config, "youtube", None)
    metadata = _build_metadata(
        title,
        caption,
        privacy_status=(
            str(youtube_settings.privacyStatus)
            if youtube_settings is not None
            else "public"
        ),
        self_declared_made_for_kids=(
            bool(youtube_settings.selfDeclaredMadeForKids)
            if youtube_settings is not None
            else False
        ),
    )

    logger.info("Initiating YouTube resumable upload for publication %s (%d bytes)", publication.id, media.file_size)

    upload_url = _initiate_resumable_upload(
        access_token=account.tokens.access_token,
        metadata=metadata,
        file_size=media.file_size,
        content_type=media.content_type,
        notify_subscribers=(
            bool(youtube_settings.notifySubscribers)
            if youtube_settings is not None
            else True
        ),
    )

    logger.info("Uploading video bytes for publication %s", publication.id)

    payload = _upload_video_bytes(
        upload_url=upload_url,
        local_path=media.local_path,
        file_size=media.file_size,
        content_type=media.content_type,
    )

    video_id = payload.get("id") if isinstance(payload, dict) else None
    if not video_id:
        raise SocialProviderError(
            "YouTube upload succeeded without returning a video id.",
            code="youtube_missing_video_id",
            provider_payload=payload,
        )

    logger.info("YouTube upload complete: video_id=%s", video_id)

    return PublicationResult(
        remote_post_id=str(video_id),
        remote_post_url=f"https://www.youtube.com/shorts/{video_id}",
        provider_payload=payload,
        result_payload={"platform": "youtube", "video_id": video_id},
    )

"""Provider registry for social publishing."""

from __future__ import annotations

from services.social import meta_facebook, meta_instagram, tiktok, youtube
from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
    SocialAccountContext,
)


def publish_to_provider(
    *,
    provider: str,
    account: SocialAccountContext,
    publication: PublicationContext,
    media: PublicationMedia,
) -> PublicationResult:
    normalized = str(provider or "").strip().lower()
    if normalized == "tiktok":
        return tiktok.publish_video(account=account, publication=publication, media=media)
    if normalized == "facebook_page":
        return meta_facebook.publish_video(account=account, publication=publication, media=media)
    if normalized == "instagram_business":
        return meta_instagram.publish_reel(account=account, publication=publication, media=media)
    if normalized == "youtube_channel":
        return youtube.publish_video(account=account, publication=publication, media=media)
    raise ValueError(f"Unsupported social provider: {provider}")

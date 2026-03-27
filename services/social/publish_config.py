"""Typed publish-config models shared by API and workers."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PublishProvider = Literal[
    "tiktok",
    "youtube_channel",
    "instagram_business",
    "facebook_page",
]
TikTokPrivacyLevel = Literal[
    "PUBLIC_TO_EVERYONE",
    "MUTUAL_FOLLOW_FRIENDS",
    "FOLLOWER_OF_CREATOR",
    "SELF_ONLY",
]
YouTubePrivacyStatus = Literal["public", "private", "unlisted"]


class PublishSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["now", "schedule"]
    scheduledAt: datetime | None = None
    timeZone: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_schedule(self) -> "PublishSchedule":
        if self.mode == "schedule" and self.scheduledAt is None:
            raise ValueError("scheduledAt is required when mode=schedule")
        return self


class PublishContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=100)
    caption: str | None = Field(default=None, max_length=5000)
    coverTimestampMs: int | None = Field(default=None, ge=0)


class TikTokCreatorInfoSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creatorNickname: str = Field(..., min_length=1)
    creatorUsername: str = Field(..., min_length=1)
    creatorAvatarUrl: str | None = None
    privacyLevelOptions: list[TikTokPrivacyLevel] = Field(..., min_length=1)
    commentDisabled: bool
    duetDisabled: bool
    stitchDisabled: bool
    maxVideoPostDurationSec: int | None = Field(default=None, ge=1)
    fetchedAt: datetime
    canPost: bool
    blockedReason: str | None = None


class TikTokPublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creatorInfo: TikTokCreatorInfoSnapshot
    privacyLevel: TikTokPrivacyLevel
    allowComment: bool = False
    allowDuet: bool = False
    allowStitch: bool = False
    commercialContentEnabled: bool = False
    brandOrganicToggle: bool = False
    brandContentToggle: bool = False
    declarationAccepted: bool

    @model_validator(mode="after")
    def validate_tiktok_settings(self) -> "TikTokPublishSettings":
        if not self.creatorInfo.canPost:
            raise ValueError(
                self.creatorInfo.blockedReason
                or "This TikTok account cannot publish right now."
            )

        if self.privacyLevel not in self.creatorInfo.privacyLevelOptions:
            raise ValueError(
                "TikTok privacyLevel must match a value returned by creator_info/query."
            )

        if self.allowComment and self.creatorInfo.commentDisabled:
            raise ValueError(
                "Comments are disabled for this TikTok account."
            )
        if self.allowDuet and self.creatorInfo.duetDisabled:
            raise ValueError("Duet is unavailable for this TikTok account.")
        if self.allowStitch and self.creatorInfo.stitchDisabled:
            raise ValueError("Stitch is unavailable for this TikTok account.")

        if self.commercialContentEnabled:
            if not self.brandOrganicToggle and not self.brandContentToggle:
                raise ValueError(
                    "Commercial content requires selecting Your Brand, Branded Content, or both."
                )
            if self.brandContentToggle and self.privacyLevel == "SELF_ONLY":
                raise ValueError(
                    "Branded content visibility cannot be set to private."
                )

        if not self.declarationAccepted:
            raise ValueError("TikTok declaration must be accepted before publishing.")

        return self


class YouTubePublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    privacyStatus: YouTubePrivacyStatus = "unlisted"
    notifySubscribers: bool = False
    selfDeclaredMadeForKids: bool = False


class InstagramPublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: Literal["account_default"] = "account_default"


class FacebookPublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: Literal["public"] = "public"


class BasePublishDestinationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clipId: str = Field(..., min_length=1)
    socialAccountId: str = Field(..., min_length=1)
    provider: PublishProvider
    schedule: PublishSchedule
    content: PublishContent


class TikTokPublishDestinationConfig(BasePublishDestinationConfig):
    provider: Literal["tiktok"]
    tiktok: TikTokPublishSettings

    @model_validator(mode="after")
    def validate_tiktok_content(self) -> "TikTokPublishDestinationConfig":
        if self.content.caption is not None and len(self.content.caption) > 2200:
            raise ValueError("TikTok captions must be 2200 characters or fewer.")
        return self


class YouTubePublishDestinationConfig(BasePublishDestinationConfig):
    provider: Literal["youtube_channel"]
    youtube: YouTubePublishSettings

    @model_validator(mode="after")
    def validate_youtube_content(self) -> "YouTubePublishDestinationConfig":
        title = (self.content.title or "").strip()
        if not title:
            raise ValueError("YouTube publishing requires a title.")
        return self


class InstagramPublishDestinationConfig(BasePublishDestinationConfig):
    provider: Literal["instagram_business"]
    instagram: InstagramPublishSettings


class FacebookPublishDestinationConfig(BasePublishDestinationConfig):
    provider: Literal["facebook_page"]
    facebook: FacebookPublishSettings


PublishDestinationConfig = Annotated[
    (
        TikTokPublishDestinationConfig
        | YouTubePublishDestinationConfig
        | InstagramPublishDestinationConfig
        | FacebookPublishDestinationConfig
    ),
    Field(discriminator="provider"),
]


def destination_schedule_identity(config: PublishDestinationConfig) -> tuple[str, datetime | None, str]:
    return (
        config.schedule.mode,
        config.schedule.scheduledAt,
        config.schedule.timeZone,
    )


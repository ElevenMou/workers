"""Typed publish-config models shared by API and workers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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
YouTubeLicense = Literal["creativeCommon", "youtube"]
InstagramTrialGraduationStrategy = Literal["MANUAL", "SS_PERFORMANCE"]
FacebookPublishTarget = Literal["auto", "reel", "page_video"]
FacebookContentCategory = Literal[
    "BEAUTY_FASHION",
    "BUSINESS",
    "CARS_TRUCKS",
    "COMEDY",
    "CUTE_ANIMALS",
    "ENTERTAINMENT",
    "FAMILY",
    "FOOD_HEALTH",
    "HOME",
    "LIFESTYLE",
    "MUSIC",
    "NEWS",
    "POLITICS",
    "SCIENCE",
    "SPORTS",
    "TECHNOLOGY",
    "VIDEO_GAMING",
    "OTHER",
]


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
    creatorAvatarUrl: HttpUrl | None = None
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
    isAigc: bool = False

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
            raise ValueError("Comments are disabled for this TikTok account.")
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
    license: YouTubeLicense = "youtube"
    embeddable: bool = True
    publicStatsViewable: bool = True
    categoryId: str = Field(default="22", min_length=1, max_length=12)
    tags: list[str] = Field(default_factory=list)
    defaultLanguage: str | None = Field(default=None, min_length=2, max_length=35)
    recordingDate: date | None = None
    containsSyntheticMedia: bool = False
    hasPaidProductPlacement: bool = False
    customThumbnailUrl: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_tags(self) -> "YouTubePublishSettings":
        combined_length = len(",".join(self.tags))
        if combined_length > 500:
            raise ValueError("YouTube tags must stay within 500 total characters.")
        return self


class InstagramUserTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, max_length=64)


class InstagramPublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: Literal["account_default"] = "account_default"
    audioName: str | None = Field(default=None, max_length=255)
    collaborators: list[str] = Field(default_factory=list, max_length=3)
    coverUrl: HttpUrl | None = None
    locationId: str | None = Field(default=None, max_length=255)
    thumbOffsetSeconds: float | None = Field(default=None, ge=0)
    userTags: list[InstagramUserTag] = Field(default_factory=list)
    commentsEnabled: bool = True
    trialEnabled: bool = False
    trialGraduationStrategy: InstagramTrialGraduationStrategy | None = None

    @model_validator(mode="after")
    def validate_trial_settings(self) -> "InstagramPublishSettings":
        if self.trialEnabled and self.trialGraduationStrategy is None:
            raise ValueError("Trial reels require a trialGraduationStrategy.")
        return self


class FacebookNumericKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: int = Field(..., gt=0)


class FacebookZipKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=32)


class FacebookGeoLocations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    countries: list[str] = Field(default_factory=list)
    regions: list[FacebookNumericKey] = Field(default_factory=list)
    cities: list[FacebookNumericKey] = Field(default_factory=list)
    zips: list[FacebookZipKey] = Field(default_factory=list)


class FacebookFeedTargeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geoLocations: FacebookGeoLocations = Field(default_factory=FacebookGeoLocations)
    locales: list[int] = Field(default_factory=list)
    ageMin: int | None = Field(default=None, ge=13)
    ageMax: int | None = Field(default=None, ge=13)
    genders: list[Literal[1, 2]] = Field(default_factory=list)
    collegeYears: list[int] = Field(default_factory=list)
    educationStatuses: list[Literal[1, 2, 3]] = Field(default_factory=list)
    relationshipStatuses: list[Literal[1, 2, 3, 4]] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)


class FacebookTargeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geoLocations: FacebookGeoLocations = Field(default_factory=FacebookGeoLocations)
    locales: list[int] = Field(default_factory=list)
    ageMin: Literal[13, 15, 18, 21, 25] | None = None
    ageMax: int | None = Field(default=None, ge=13)
    genders: list[Literal[1, 2]] = Field(default_factory=list)
    collegeYears: list[int] = Field(default_factory=list)
    educationStatuses: list[Literal[1, 2, 3]] = Field(default_factory=list)
    relationshipStatuses: list[Literal[1, 2, 3, 4]] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    excludedCountries: list[str] = Field(default_factory=list)
    excludedRegions: list[FacebookNumericKey] = Field(default_factory=list)
    excludedCities: list[FacebookNumericKey] = Field(default_factory=list)
    excludedZipcodes: list[FacebookZipKey] = Field(default_factory=list)
    timezones: list[int] = Field(default_factory=list)


class FacebookPublishSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: Literal["public"] = "public"
    publishTarget: FacebookPublishTarget = "auto"
    placeId: str | None = Field(default=None, max_length=255)
    contentCategory: FacebookContentCategory | None = None
    contentTags: list[str] = Field(default_factory=list)
    hideFromNewsfeed: bool = False
    collaboratorPageId: str | None = Field(default=None, max_length=255)
    crosspostPageIds: list[str] = Field(default_factory=list)
    allowBusinessManagerCrossposting: bool = False
    feedTargeting: FacebookFeedTargeting | None = None
    targeting: FacebookTargeting | None = None


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


def destination_schedule_identity(
    config: PublishDestinationConfig,
) -> tuple[str, datetime | None, str]:
    return (
        config.schedule.mode,
        config.schedule.scheduledAt,
        config.schedule.timeZone,
    )


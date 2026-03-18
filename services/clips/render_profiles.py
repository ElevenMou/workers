"""Render profile helpers for source, master, delivery, and publishing."""

from __future__ import annotations

from typing import Any

from services.media_profiles import clamped_source_fps, is_hdr_profile

DEFAULT_SOURCE_PROFILE = "source_best_2160"
DEFAULT_MASTER_PROFILE = "prores_422hq_archive"
DEFAULT_DELIVERY_PROFILE = "social_auto_h264"
DEFAULT_PUBLISH_PROFILE = "browser_delivery_h264"

_DELIVERY_PROFILE_BY_ASPECT = {
    "9:16": "vertical_social_1080x1920",
    "4:5": "feed_social_1080x1350",
    "1:1": "square_social_1080x1080",
    "16:9": "landscape_social_1920x1080",
}


def source_profile_max_height(source_profile: str | None) -> int | None:
    normalized = str(source_profile or "").strip().lower()
    if "2160" in normalized or normalized.endswith("best"):
        return 2160
    if "1440" in normalized:
        return 1440
    if "1080" in normalized:
        return 1080
    if "720" in normalized:
        return 720
    return 2160


def delivery_profile_for_aspect_ratio(canvas_aspect_ratio: str | None) -> str:
    normalized = str(canvas_aspect_ratio or "").strip()
    return _DELIVERY_PROFILE_BY_ASPECT.get(normalized, _DELIVERY_PROFILE_BY_ASPECT["9:16"])


def publish_profile_for_provider(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "youtube_channel":
        return "youtube_h264_upload"
    if normalized == "instagram_business":
        return "instagram_reels_h264_upload"
    if normalized == "facebook_page":
        return "facebook_reels_h264_upload"
    if normalized == "tiktok":
        return "tiktok_h264_upload"
    return DEFAULT_PUBLISH_PROFILE


def build_master_encode_args(*, source_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    encode_args: dict[str, Any] = {
        "vcodec": "prores_ks",
        "acodec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "profile:v": 3,
        "ar": 48000,
        "ac": 2,
    }
    if isinstance(source_profile, dict) and not is_hdr_profile(source_profile):
        video_profile = source_profile.get("video") or {}
        for profile_key, ffmpeg_key in (
            ("color_space", "colorspace"),
            ("color_primaries", "color_primaries"),
            ("color_transfer", "color_trc"),
        ):
            value = str(video_profile.get(profile_key) or "").strip()
            if value:
                encode_args[ffmpeg_key] = value
        color_range = str(video_profile.get("color_range") or "").strip()
        if color_range:
            encode_args["color_range"] = color_range
    return encode_args


def build_delivery_encode_args(
    *,
    delivery_profile: str,
    source_profile: dict[str, Any] | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    fps = clamped_source_fps(source_profile, fallback=30.0, max_fps=60.0)
    fps_is_high = fps > 30.5
    gop = int(round(fps * 2.0))
    maxrate = "20M" if fps_is_high else "16M"
    bufsize = "40M" if fps_is_high else "32M"
    level = "4.2" if fps_is_high else "4.1"
    crf = 14 if "youtube" in str(profile_name or delivery_profile).lower() else 15
    return {
        "vcodec": "libx264",
        "acodec": "aac",
        "crf": crf,
        "preset": "slow",
        "profile:v": "high",
        "level": level,
        "tune": "film",
        "maxrate": maxrate,
        "bufsize": bufsize,
        "g": gop,
        "keyint_min": gop,
        "pix_fmt": "yuv420p",
        "audio_bitrate": "320k",
        "ar": 48000,
        "ac": 2,
        "movflags": "+faststart",
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
    }


def requires_sdr_tonemap(source_profile: dict[str, Any] | None) -> bool:
    return is_hdr_profile(source_profile)


__all__ = [
    "DEFAULT_DELIVERY_PROFILE",
    "DEFAULT_MASTER_PROFILE",
    "DEFAULT_PUBLISH_PROFILE",
    "DEFAULT_SOURCE_PROFILE",
    "build_delivery_encode_args",
    "build_master_encode_args",
    "delivery_profile_for_aspect_ratio",
    "publish_profile_for_provider",
    "requires_sdr_tonemap",
    "source_profile_max_height",
]

"""Shared FFprobe-based media profile helpers."""

from __future__ import annotations

from typing import Any

import ffmpeg

_HDR_COLOR_PRIMARIES = {"bt2020", "bt2020nc", "bt2020ncl"}
_HDR_COLOR_TRANSFER = {"smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12"}


def _safe_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_frame_rate(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except (TypeError, ValueError):
            return None
        if den == 0:
            return None
        return num / den
    return _safe_float(raw)


def probe_media_profile(path: str) -> dict[str, Any]:
    probe = ffmpeg.probe(path)
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    video_stream = next(
        (stream for stream in streams if str(stream.get("codec_type")) == "video"),
        {},
    )
    audio_stream = next(
        (stream for stream in streams if str(stream.get("codec_type")) == "audio"),
        {},
    )
    fps = (
        parse_frame_rate(video_stream.get("avg_frame_rate"))
        or parse_frame_rate(video_stream.get("r_frame_rate"))
    )
    return {
        "format_name": format_info.get("format_name"),
        "format_long_name": format_info.get("format_long_name"),
        "duration": _safe_float(format_info.get("duration")),
        "bit_rate": _safe_int(format_info.get("bit_rate")),
        "video": {
            "codec_name": video_stream.get("codec_name"),
            "codec_tag_string": video_stream.get("codec_tag_string"),
            "width": _safe_int(video_stream.get("width")),
            "height": _safe_int(video_stream.get("height")),
            "bit_rate": _safe_int(video_stream.get("bit_rate")),
            "fps": fps,
            "r_frame_rate": video_stream.get("r_frame_rate"),
            "avg_frame_rate": video_stream.get("avg_frame_rate"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "color_primaries": video_stream.get("color_primaries"),
            "color_transfer": video_stream.get("color_transfer"),
            "color_space": video_stream.get("color_space"),
            "color_range": video_stream.get("color_range"),
        },
        "audio": {
            "codec_name": audio_stream.get("codec_name"),
            "bit_rate": _safe_int(audio_stream.get("bit_rate")),
            "sample_rate": _safe_int(audio_stream.get("sample_rate")),
            "channels": _safe_int(audio_stream.get("channels")),
            "channel_layout": audio_stream.get("channel_layout"),
        },
    }


def build_video_source_probe_update(path: str) -> dict[str, Any]:
    profile = probe_media_profile(path)
    video_profile = profile.get("video") or {}
    audio_profile = profile.get("audio") or {}
    return {
        "source_container": profile.get("format_name"),
        "source_video_codec": video_profile.get("codec_name"),
        "source_audio_codec": audio_profile.get("codec_name"),
        "source_width": video_profile.get("width"),
        "source_height": video_profile.get("height"),
        "source_fps": video_profile.get("fps"),
        "source_video_bitrate": video_profile.get("bit_rate"),
        "source_audio_bitrate": audio_profile.get("bit_rate"),
        "source_color_primaries": video_profile.get("color_primaries"),
        "source_color_transfer": video_profile.get("color_transfer"),
        "source_color_space": video_profile.get("color_space"),
        "source_color_range": video_profile.get("color_range"),
        "source_sample_rate": audio_profile.get("sample_rate"),
        "source_channel_layout": audio_profile.get("channel_layout"),
    }


def is_hdr_profile(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    video_profile = profile.get("video") or {}
    color_primaries = str(video_profile.get("color_primaries") or "").strip().lower()
    color_transfer = str(video_profile.get("color_transfer") or "").strip().lower()
    if color_primaries in _HDR_COLOR_PRIMARIES:
        return True
    if color_transfer in _HDR_COLOR_TRANSFER:
        return True
    return False


def clamped_source_fps(profile: dict[str, Any] | None, *, fallback: float = 30.0, max_fps: float = 60.0) -> float:
    if not isinstance(profile, dict):
        return fallback
    video_profile = profile.get("video") or {}
    fps = _safe_float(video_profile.get("fps"))
    if fps is None or fps <= 0:
        return fallback
    return min(max_fps, max(1.0, fps))


__all__ = [
    "build_video_source_probe_update",
    "clamped_source_fps",
    "is_hdr_profile",
    "parse_frame_rate",
    "probe_media_profile",
]

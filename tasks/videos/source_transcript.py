"""Shared source-transcript resolution for analysis and clip generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from services.video_downloader import VideoDownloader

logger = logging.getLogger(__name__)


def _normalize_language_code(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return text.replace("_", "-")


def _build_preferred_languages(language_hint: str | None) -> list[str]:
    preferred: list[str] = []
    normalized = _normalize_language_code(language_hint)
    if normalized and normalized not in preferred:
        preferred.append(normalized)
    if normalized:
        base = normalized.split("-", 1)[0]
        if base and base not in preferred:
            preferred.append(base)
    for fallback in ("en", "en-us", "en-gb"):
        if fallback not in preferred:
            preferred.append(fallback)
    return preferred


def _has_segments(transcript: dict[str, Any] | None) -> bool:
    return isinstance(transcript, dict) and bool(transcript.get("segments"))


@dataclass(slots=True)
class TranscriptResolution:
    transcript: dict[str, Any]
    is_full_transcript: bool
    diagnostics: dict[str, Any]


def resolve_source_transcript(
    *,
    existing_transcript: dict[str, Any] | None,
    downloader: VideoDownloader,
    source_url: str,
    source_platform: str | None,
    source_external_id: str | None,
    source_detected_language: str | None,
    source_has_audio: bool | None,
    whisper_fallback: Callable[[str | None], tuple[dict[str, Any], bool]],
    job_id: str,
    on_attempt: Callable[[str], None] | None = None,
) -> TranscriptResolution:
    preferred_languages = _build_preferred_languages(source_detected_language)
    diagnostics: dict[str, Any] = {
        "transcript_source": None,
        "transcript_language": None,
        "transcript_reused": False,
        "transcript_fallback_reason": None,
        "provider_track_source": None,
        "provider_track_ext": None,
        "provider_track_language": None,
    }

    if _has_segments(existing_transcript):
        transcript = dict(existing_transcript or {})
        diagnostics["transcript_source"] = (
            str(transcript.get("source") or "").strip().lower() or "stored_transcript"
        )
        diagnostics["transcript_language"] = (
            _normalize_language_code(
                str(transcript.get("languageCode") or transcript.get("language") or "")
            )
            or "unknown"
        )
        diagnostics["transcript_reused"] = True
        return TranscriptResolution(
            transcript=transcript,
            is_full_transcript=True,
            diagnostics=diagnostics,
        )

    normalized_platform = str(source_platform or "").strip().lower()
    source_external_id = str(source_external_id or "").strip() or None

    if normalized_platform == "youtube" and source_external_id:
        if on_attempt:
            on_attempt("youtube_captions")
        try:
            transcript = downloader.get_youtube_transcript(
                source_external_id,
                preferred_languages=preferred_languages,
            )
        except Exception as exc:
            logger.warning("[%s] YouTube transcript fetch failed: %s", job_id, exc)
            transcript = None
        if _has_segments(transcript):
            diagnostics["transcript_source"] = "youtube"
            diagnostics["transcript_language"] = (
                _normalize_language_code(
                    str(transcript.get("languageCode") or transcript.get("language") or "")
                )
                or "unknown"
            )
            return TranscriptResolution(
                transcript=dict(transcript),
                is_full_transcript=True,
                diagnostics=diagnostics,
            )

    if on_attempt:
        on_attempt("provider_captions")
    get_provider_transcript = getattr(downloader, "get_provider_transcript", None)
    if callable(get_provider_transcript):
        try:
            provider_result = get_provider_transcript(
                source_url,
                preferred_languages=preferred_languages,
            )
        except Exception as exc:
            logger.warning("[%s] Provider transcript fetch failed: %s", job_id, exc)
            provider_result = {
                "transcript": None,
                "fallback_reason": f"provider_caption_probe_failed:{type(exc).__name__}",
                "track_source": None,
                "track_ext": None,
                "track_language": None,
            }
    else:
        provider_result = {
            "transcript": None,
            "fallback_reason": "provider_caption_unavailable",
            "track_source": None,
            "track_ext": None,
            "track_language": None,
        }
    provider_transcript = provider_result.get("transcript")
    if _has_segments(provider_transcript):
        diagnostics["transcript_source"] = (
            str(provider_transcript.get("source") or "").strip().lower()
            or "provider_captions"
        )
        diagnostics["transcript_language"] = (
            _normalize_language_code(
                str(
                    provider_transcript.get("languageCode")
                    or provider_transcript.get("language")
                    or ""
                )
            )
            or "unknown"
        )
        diagnostics["provider_track_source"] = provider_result.get("track_source")
        diagnostics["provider_track_ext"] = provider_result.get("track_ext")
        diagnostics["provider_track_language"] = provider_result.get("track_language")
        return TranscriptResolution(
            transcript=dict(provider_transcript),
            is_full_transcript=True,
            diagnostics=diagnostics,
        )

    diagnostics["transcript_fallback_reason"] = provider_result.get("fallback_reason")
    diagnostics["provider_track_source"] = provider_result.get("track_source")
    diagnostics["provider_track_ext"] = provider_result.get("track_ext")
    diagnostics["provider_track_language"] = provider_result.get("track_language")

    if source_has_audio is False:
        raise RuntimeError(
            "Source does not expose downloadable audio and no source captions were found"
        )

    if on_attempt:
        on_attempt("whisper_fallback")
    transcript, is_full_transcript = whisper_fallback(source_detected_language)
    if not _has_segments(transcript):
        raise RuntimeError("Whisper fallback did not return a usable transcript")

    diagnostics["transcript_source"] = (
        str(transcript.get("source") or "").strip().lower() or "whisper"
    )
    diagnostics["transcript_language"] = (
        _normalize_language_code(
            str(transcript.get("languageCode") or transcript.get("language") or "")
        )
        or _normalize_language_code(source_detected_language)
        or "unknown"
    )
    return TranscriptResolution(
        transcript=dict(transcript),
        is_full_transcript=bool(is_full_transcript),
        diagnostics=diagnostics,
    )

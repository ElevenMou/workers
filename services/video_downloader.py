import ipaddress
import logging
import os
import socket
import time
from glob import glob
from urllib.parse import urlparse

import ffmpeg as ffmpeg_lib
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from config import (
    MAX_VIDEO_SIZE_MB,
    TEMP_DIR,
    YTDLP_DOWNLOAD_RETRIES,
    YTDLP_EXTRACTOR_RETRIES,
    YTDLP_FRAGMENT_RETRIES,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
)

_yt_transcript_api = YouTubeTranscriptApi()

logger = logging.getLogger(__name__)

# SECURITY: Only allow known video platform domains to prevent SSRF attacks
# against internal services (metadata endpoints, Redis, etc.).
_ALLOWED_DOMAINS: set[str] = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
    "player.vimeo.com",
    "dailymotion.com",
    "www.dailymotion.com",
    "twitch.tv",
    "www.twitch.tv",
    "clips.twitch.tv",
    "facebook.com",
    "www.facebook.com",
    "fb.watch",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
    "vm.tiktok.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "www.reddit.com",
    "v.redd.it",
    "streamable.com",
    "www.streamable.com",
}
_YOUTUBE_DOMAINS: set[str] = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}


def _validate_url(url: str) -> None:
    """Validate that a URL targets an allowed video platform.

    Blocks SSRF attempts against internal services, metadata endpoints,
    private IP ranges, and non-HTTP schemes.
    """
    parsed = urlparse(url)

    # Only allow HTTP(S)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL has no hostname")

    # Block IP addresses entirely (prevents private range probing)
    try:
        ip = ipaddress.ip_address(hostname)
        raise ValueError(f"IP-based URLs are not allowed: {ip}")
    except ValueError as exc:
        if "IP-based URLs are not allowed" in str(exc):
            raise

    # Also resolve the hostname and check for private IPs
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError(f"URL resolves to a private/reserved IP: {ip}")
    except socket.gaierror:
        pass  # DNS resolution failure will be caught by yt-dlp

    # Check domain allowlist
    if hostname not in _ALLOWED_DOMAINS:
        raise ValueError(
            f"Domain '{hostname}' is not in the allowed video platforms list"
        )


class VideoDownloader:
    def __init__(self, work_dir: str | None = None):
        self.temp_dir = work_dir or TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def _normalize_max_height(max_height: int | None) -> int | None:
        if max_height is None:
            return None
        try:
            parsed = int(max_height)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    @classmethod
    def _format_selector_for_max_height(cls, max_height: int | None) -> str:
        normalized_max_height = cls._normalize_max_height(max_height)
        if normalized_max_height is None:
            return (
                "bv[vcodec^=avc1]+ba[acodec^=mp4a]/"
                "bv[vcodec^=avc1]+ba/"
                "bv+ba/"
                "b/"
                "bv*+ba/b"
            )

        height_filter = f"[height<={normalized_max_height}]"
        return (
            f"bv{height_filter}[vcodec^=avc1]+ba[acodec^=mp4a]/"
            f"bv{height_filter}[vcodec^=avc1]+ba/"
            f"bv{height_filter}+ba/"
            f"b{height_filter}/"
            "bv*+ba/b"
        )

    @classmethod
    def _base_ydl_opts(cls, *, max_height: int | None = 1080) -> dict:
        return {
            # Prefer h264+AAC for fast downstream processing.
            # Height cap is controlled by tier-aware quality policy.
            "format": cls._format_selector_for_max_height(max_height),
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
            "merge_output_format": "mp4",
            "noplaylist": True,
            # Enable JS runtimes for YouTube challenge solving hardening.
            "js_runtimes": {"node": {}, "deno": {}},
            # Set a desktop UA to reduce throttling/403
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
            "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
            "retries": YTDLP_DOWNLOAD_RETRIES,
            "fragment_retries": YTDLP_FRAGMENT_RETRIES,
            "extractor_retries": YTDLP_EXTRACTOR_RETRIES,
        }

    @staticmethod
    def _selected_format_summary(info: dict) -> list[dict[str, object]]:
        selected = info.get("requested_formats")
        entries = selected if isinstance(selected, list) and selected else [info]
        summaries: list[dict[str, object]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            summaries.append(
                {
                    "format_id": entry.get("format_id"),
                    "ext": entry.get("ext"),
                    "height": entry.get("height"),
                    "vcodec": entry.get("vcodec"),
                    "acodec": entry.get("acodec"),
                    "fps": entry.get("fps"),
                }
            )

        return summaries

    @staticmethod
    def _format_summary_text(summaries: list[dict[str, object]]) -> str:
        if not summaries:
            return "unknown"

        tokens: list[str] = []
        for summary in summaries:
            tokens.append(
                "id={format_id} ext={ext} height={height} fps={fps} vcodec={vcodec} acodec={acodec}".format(
                    format_id=summary.get("format_id") or "-",
                    ext=summary.get("ext") or "-",
                    height=summary.get("height") or "-",
                    fps=summary.get("fps") or "-",
                    vcodec=summary.get("vcodec") or "-",
                    acodec=summary.get("acodec") or "-",
                )
            )
        return "; ".join(tokens)

    @staticmethod
    def _has_audio_stream(video_path: str) -> bool:
        try:
            probe = ffmpeg_lib.probe(video_path)
        except Exception as exc:
            logger.warning("Failed to probe audio streams in %s: %s", video_path, exc)
            return False
        return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))

    @staticmethod
    def _resolve_output_path(expected_path: str) -> str:
        if os.path.isfile(expected_path):
            return expected_path

        base, _ = os.path.splitext(expected_path)
        candidates = [p for p in glob(f"{base}.*") if os.path.isfile(p)]
        if not candidates:
            return expected_path
        return max(candidates, key=os.path.getmtime)

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        hostname = (urlparse(url).hostname or "").strip().lower()
        return hostname in _YOUTUBE_DOMAINS

    def _download_with_opts(
        self,
        url: str,
        output_path: str,
        *,
        format_selector: str,
        attempt_label: str,
        max_height: int | None = 1080,
        ydl_overrides: dict[str, object] | None = None,
    ) -> dict:
        ydl_opts = self._base_ydl_opts(max_height=max_height)
        ydl_opts.update(
            {
                "format": format_selector,
                "outtmpl": output_path,
                "quiet": False,
                "no_warnings": False,
            }
        )
        if ydl_overrides:
            ydl_opts.update(ydl_overrides)
        started_at = time.monotonic()
        video_hint = os.path.splitext(os.path.basename(output_path))[0]
        logger.info(
            "yt-dlp download start (%s): video=%s timeout=%ss retries=%s/%s/%s",
            attempt_label,
            video_hint,
            ydl_opts.get("socket_timeout"),
            ydl_opts.get("retries"),
            ydl_opts.get("fragment_retries"),
            ydl_opts.get("extractor_retries"),
        )
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                logger.info(
                    "yt-dlp selected formats (%s): %s",
                    attempt_label,
                    self._format_summary_text(self._selected_format_summary(info)),
                )
                return info
        finally:
            elapsed = time.monotonic() - started_at
            logger.info(
                "yt-dlp download finished (%s): video=%s elapsed=%.2fs",
                attempt_label,
                video_hint,
                elapsed,
            )

    def download(self, url: str, video_id: str, *, max_height: int | None = 1080) -> dict:
        """Download video and return metadata."""
        _validate_url(url)
        output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")
        normalized_max_height = self._normalize_max_height(max_height)
        primary_selector = self._format_selector_for_max_height(normalized_max_height)
        legacy_client_overrides = {
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}}
        }
        try:
            info = self._download_with_opts(
                url,
                output_path,
                format_selector=primary_selector,
                attempt_label="primary_av_merge_default",
                max_height=normalized_max_height,
            )
        except yt_dlp.utils.DownloadError as exc:
            if not self._is_youtube_url(url):
                raise
            logger.warning(
                "yt-dlp default-client download failed for %s; retrying with legacy YouTube clients: %s",
                video_id,
                exc,
            )
            info = self._download_with_opts(
                url,
                output_path,
                format_selector=primary_selector,
                attempt_label="primary_av_merge_legacy_clients",
                max_height=normalized_max_height,
                ydl_overrides=legacy_client_overrides,
            )
        downloaded_path = self._resolve_output_path(output_path)

        if not self._has_audio_stream(downloaded_path):
            logger.warning(
                "Downloaded media for %s has no audio stream; retrying with muxed A/V format",
                video_id,
            )
            try:
                os.remove(downloaded_path)
            except OSError:
                pass

            info = self._download_with_opts(
                url,
                output_path,
                format_selector=(
                    f"b[ext=mp4][vcodec!=none][acodec!=none]"
                    f"{f'[height<={normalized_max_height}]' if normalized_max_height else ''}/"
                    f"b[vcodec!=none][acodec!=none]"
                    f"{f'[height<={normalized_max_height}]' if normalized_max_height else ''}/"
                    "b[vcodec!=none][acodec!=none]"
                ),
                attempt_label="fallback_muxed_av",
                max_height=normalized_max_height,
            )
            downloaded_path = self._resolve_output_path(output_path)

        if not self._has_audio_stream(downloaded_path):
            logger.warning("Downloaded media for %s still has no audio stream", video_id)

        return {
            "path": downloaded_path,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "platform": info.get("extractor_key", "unknown").lower(),
            "external_id": info.get("id"),
        }

    def probe_url(self, url: str) -> dict:
        """Validate URL with yt-dlp and return probe metadata.

        This does not download media. It only extracts metadata needed for:
        - can we access downloadable formats?
        - does the source expose subtitles/captions?
        - what is the duration for cost calculation?
        """
        _validate_url(url)
        ydl_opts = self._base_ydl_opts(max_height=1080)
        ydl_opts.update(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
        )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        duration_seconds = int(info.get("duration") or 0)
        subtitles = info.get("subtitles") or {}
        automatic_captions = info.get("automatic_captions") or {}

        has_captions = bool(subtitles) or bool(automatic_captions)
        formats = info.get("formats") or []
        can_download = bool(formats) or bool(info.get("url"))
        has_audio = any((f.get("acodec") not in (None, "none")) for f in formats)
        detected_language = str(info.get("language") or "").strip() or None
        if not detected_language:
            language_candidates = list(subtitles.keys()) + list(automatic_captions.keys())
            if language_candidates:
                detected_language = str(language_candidates[0]).strip() or None

        return {
            "can_download": can_download,
            "has_captions": has_captions,
            "has_audio": has_audio,
            "duration_seconds": duration_seconds,
            "platform": info.get("extractor_key", "unknown").lower(),
            "external_id": info.get("id"),
            "title": info.get("title"),
            "thumbnail": info.get("thumbnail"),
            "detected_language": detected_language,
        }

    def get_youtube_transcript(self, video_id: str) -> dict | None:
        """Get transcript directly from YouTube (v1.x API).

        Tries English first, then falls back to any available language.
        """
        fetched = None

        # 1. Try English transcript (manual or auto-generated)
        try:
            fetched = _yt_transcript_api.fetch(
                video_id, languages=["en", "en-US", "en-GB"]
            )
            logger.info(
                "Found %s transcript (%s) for %s",
                fetched.language,
                fetched.language_code,
                video_id,
            )
        except Exception:
            pass

        # 2. Fall back to any available language
        if fetched is None:
            try:
                fetched = _yt_transcript_api.fetch(video_id)
                logger.info(
                    "Found %s transcript (%s) for %s",
                    fetched.language,
                    fetched.language_code,
                    video_id,
                )
            except Exception as e:
                logger.warning("Could not get YouTube transcript for %s: %s", video_id, e)
                return None

        segments = [
            {
                "id": i,
                "start": snippet.start,
                "end": snippet.start + snippet.duration,
                "text": snippet.text,
            }
            for i, snippet in enumerate(fetched)
        ]

        full_text = " ".join(s["text"] for s in segments)

        return {
            "text": full_text,
            "segments": segments,
            "source": "youtube",
            "language": fetched.language_code,
            "languageCode": fetched.language_code,
        }

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video for Whisper transcription."""
        import ffmpeg

        audio_path = video_path.replace(".mp4", ".wav")

        (
            ffmpeg.input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16k")
            .overwrite_output()
            .run(quiet=True)
        )

        return audio_path


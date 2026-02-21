import ipaddress
import logging
import os
import socket
from glob import glob
from urllib.parse import urlparse

import ffmpeg as ffmpeg_lib
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from config import TEMP_DIR, MAX_VIDEO_SIZE_MB

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
    def _base_ydl_opts() -> dict:
        return {
            # Prefer separate A/V streams (merged locally), then muxed fallback.
            # This avoids silent "video-only" files for sources with fragmented media.
            "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
            "merge_output_format": "mp4",
            # Try multiple YouTube client profiles to avoid signature/403 issues
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
            # Set a desktop UA to reduce throttling/403
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
            "retries": 3,
            "fragment_retries": 3,
        }

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

    def _download_with_opts(self, url: str, output_path: str, *, format_selector: str) -> dict:
        ydl_opts = self._base_ydl_opts()
        ydl_opts.update(
            {
                "format": format_selector,
                "outtmpl": output_path,
                "quiet": False,
                "no_warnings": False,
            }
        )
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=True)

    def download(self, url: str, video_id: str) -> dict:
        """Download video and return metadata."""
        _validate_url(url)
        output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")

        info = self._download_with_opts(
            url,
            output_path,
            format_selector="bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
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
                    "b[ext=mp4][height<=720][vcodec!=none][acodec!=none]/"
                    "b[height<=720][vcodec!=none][acodec!=none]/"
                    "b[vcodec!=none][acodec!=none]"
                ),
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
        ydl_opts = self._base_ydl_opts()
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

        return {
            "can_download": can_download,
            "has_captions": has_captions,
            "has_audio": has_audio,
            "duration_seconds": duration_seconds,
            "platform": info.get("extractor_key", "unknown").lower(),
            "external_id": info.get("id"),
            "title": info.get("title"),
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

        return {"text": full_text, "segments": segments, "source": "youtube"}

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

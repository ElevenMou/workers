import ipaddress
import logging
import os
import shutil
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
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_SOURCE_FILE,
    YTDLP_DOWNLOAD_RETRIES,
    YTDLP_EXTRACTOR_RETRIES,
    YTDLP_FRAGMENT_RETRIES,
    YTDLP_POT_PROVIDER_URL,
    YTDLP_PROXY,
    YTDLP_PROXY_LIST,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
)

_MIN_FREE_DISK_MB = int(os.getenv("MIN_FREE_DISK_MB", str(MAX_VIDEO_SIZE_MB * 2 + 500)))

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
_YOUTUBE_BOT_CHECK_MESSAGES = (
    "sign in to confirm you’re not a bot",
    "sign in to confirm you’re not a bot",
)


def _build_proxy_list() -> list[str]:
    """Build an ordered list of proxies to try (primary first, then pool)."""
    proxies: list[str] = []
    if YTDLP_PROXY:
        proxies.append(YTDLP_PROXY)
    for p in YTDLP_PROXY_LIST:
        if p not in proxies:
            proxies.append(p)
    return proxies


class VideoProbeError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.failure_reason = reason


def classify_yt_dlp_error(exc: Exception) -> str:
    message = str(exc).strip().lower()
    if "read-only file system" in message and "cookie" in message:
        return "cookiefile_unwritable"
    if any(token in message for token in _YOUTUBE_BOT_CHECK_MESSAGES):
        return "youtube_bot_check"
    if "sign in" in message and "cookie" in message:
        return "youtube_auth_required"
    return "extractor_failure"


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

    def _default_runtime_cookiefile(self) -> str:
        return os.path.join(self.temp_dir, "yt-dlp", "cookies.txt")

    @staticmethod
    def _is_readable_cookie_path(path: str | None) -> bool:
        if not path:
            return False
        return os.path.isfile(path) and os.access(path, os.R_OK)

    @staticmethod
    def _is_writable_cookie_path(path: str | None) -> bool:
        if not path:
            return False
        return os.path.isfile(path) and os.access(path, os.W_OK)

    @staticmethod
    def _prepare_runtime_cookie_target(path: str | None) -> str | None:
        if not path:
            return None
        target_dir = os.path.dirname(path) or "."
        try:
            os.makedirs(target_dir, exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8"):
                    pass
        except OSError as exc:
            logger.warning(
                "yt-dlp cookie runtime path unavailable (%s): %s",
                path,
                exc,
            )
            return None

        if (
            not VideoDownloader._is_readable_cookie_path(path)
            or not VideoDownloader._is_writable_cookie_path(path)
        ):
            logger.warning(
                "yt-dlp cookiefile_unwritable: runtime cookie path is not read/write accessible: %s",
                path,
            )
            return None
        return path

    def _resolve_cookiefile(self) -> str | None:
        runtime_cookie_path = YTDLP_COOKIES_FILE
        source_cookie_path = YTDLP_COOKIES_SOURCE_FILE

        if (
            runtime_cookie_path
            and not source_cookie_path
            and self._is_readable_cookie_path(runtime_cookie_path)
            and not self._is_writable_cookie_path(runtime_cookie_path)
        ):
            source_cookie_path = runtime_cookie_path
            runtime_cookie_path = self._default_runtime_cookiefile()
            logger.warning(
                "yt-dlp cookiefile_unwritable: configured cookie path %s is read-only; using writable copy %s",
                source_cookie_path,
                runtime_cookie_path,
            )

        if source_cookie_path:
            if not self._is_readable_cookie_path(source_cookie_path):
                logger.warning(
                    "yt-dlp cookie source is not readable: %s; continuing without cookies",
                    source_cookie_path,
                )
                return None

            runtime_target = self._prepare_runtime_cookie_target(
                runtime_cookie_path or self._default_runtime_cookiefile()
            )
            if runtime_target is None:
                logger.warning(
                    "yt-dlp cookiefile_unwritable: could not prepare writable runtime cookie path; continuing without cookies",
                )
                return None

            if os.path.abspath(source_cookie_path) == os.path.abspath(runtime_target):
                return runtime_target

            try:
                shutil.copyfile(source_cookie_path, runtime_target)
            except OSError as exc:
                logger.warning(
                    "yt-dlp cookie copy failed from %s to %s: %s; continuing without cookies",
                    source_cookie_path,
                    runtime_target,
                    exc,
                )
                return None

            return runtime_target

        if runtime_cookie_path and self._is_readable_cookie_path(runtime_cookie_path):
            if self._is_writable_cookie_path(runtime_cookie_path):
                return runtime_cookie_path
            logger.warning(
                "yt-dlp cookiefile_unwritable: configured runtime cookie path is not writable: %s; continuing without cookies",
                runtime_cookie_path,
            )
            return None

        if runtime_cookie_path or source_cookie_path:
            logger.warning(
                "yt-dlp cookies unavailable: source=%s runtime=%s; continuing without cookies",
                source_cookie_path,
                runtime_cookie_path,
            )
        return None

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

    def _base_ydl_opts(self, *, max_height: int | None = 1080) -> dict:
        opts: dict = {
            # Prefer h264+AAC for fast downstream processing.
            # Height cap is controlled by tier-aware quality policy.
            "format": self._format_selector_for_max_height(max_height),
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
            "merge_output_format": "mp4",
            "noplaylist": True,
            # Let yt-dlp use its built-in default User-Agent (keeps Chrome
            # version current and avoids stale-UA bot detection).
            "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
            "retries": YTDLP_DOWNLOAD_RETRIES,
            "fragment_retries": YTDLP_FRAGMENT_RETRIES,
            "extractor_retries": YTDLP_EXTRACTOR_RETRIES,
        }
        cookiefile = self._resolve_cookiefile()
        if cookiefile:
            opts["cookiefile"] = cookiefile
        proxies = _build_proxy_list()
        if proxies:
            opts["proxy"] = proxies[0]
        if YTDLP_POT_PROVIDER_URL:
            opts.setdefault("extractor_args", {})["youtubepot-bgutilhttp"] = {
                "base_url": [YTDLP_POT_PROVIDER_URL],
            }
            logger.info(
                "yt-dlp PO token provider configured: %s",
                YTDLP_POT_PROVIDER_URL,
            )
        return opts

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
            # Deep-merge extractor_args so overrides don't clobber PO token config.
            override_ea = ydl_overrides.get("extractor_args")
            ydl_opts.update({k: v for k, v in ydl_overrides.items() if k != "extractor_args"})
            if isinstance(override_ea, dict):
                ydl_opts.setdefault("extractor_args", {}).update(override_ea)
        started_at = time.monotonic()
        video_hint = os.path.splitext(os.path.basename(output_path))[0]
        logger.info(
            "yt-dlp download start (%s): video=%s timeout=%ss retries=%s/%s/%s proxy=%s",
            attempt_label,
            video_hint,
            ydl_opts.get("socket_timeout"),
            ydl_opts.get("retries"),
            ydl_opts.get("fragment_retries"),
            ydl_opts.get("extractor_retries"),
            bool(ydl_opts.get("proxy")),
        )
        proxies = _build_proxy_list()
        last_exc: Exception | None = None
        # Try with current proxy, then rotate on bot-check failures.
        attempts = [ydl_opts.get("proxy")] + [
            p for p in proxies if p != ydl_opts.get("proxy")
        ]
        for attempt_idx, proxy in enumerate(attempts):
            if proxy:
                ydl_opts["proxy"] = proxy
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    logger.info(
                        "yt-dlp selected formats (%s): %s",
                        attempt_label,
                        self._format_summary_text(self._selected_format_summary(info)),
                    )
                    return info
            except yt_dlp.utils.DownloadError as exc:
                reason = classify_yt_dlp_error(exc)
                if reason != "youtube_bot_check" or attempt_idx >= len(attempts) - 1:
                    raise
                logger.warning(
                    "yt-dlp bot check on proxy %d/%d (%s); rotating",
                    attempt_idx + 1, len(attempts), attempt_label,
                )
                last_exc = exc
            finally:
                elapsed = time.monotonic() - started_at
                logger.info(
                    "yt-dlp download finished (%s): video=%s elapsed=%.2fs",
                    attempt_label,
                    video_hint,
                    elapsed,
                )
        raise last_exc

    @staticmethod
    def _check_disk_space(target_dir: str) -> None:
        try:
            usage = shutil.disk_usage(target_dir)
            free_mb = usage.free / (1024 * 1024)
            if free_mb < _MIN_FREE_DISK_MB:
                raise OSError(
                    f"Insufficient disk space: {free_mb:.0f} MB free, "
                    f"need at least {_MIN_FREE_DISK_MB} MB"
                )
        except OSError:
            raise
        except Exception as exc:
            logger.warning("Could not check disk space for %s: %s", target_dir, exc)

    def download(self, url: str, video_id: str, *, max_height: int | None = 1080) -> dict:
        """Download video and return metadata."""
        _validate_url(url)
        self._check_disk_space(self.temp_dir)
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

    def download_audio_only(self, url: str, video_id: str) -> dict:
        """Download source audio without fetching the full video stream."""
        _validate_url(url)
        output_template = os.path.join(self.temp_dir, f"{video_id}.audio.%(ext)s")
        legacy_client_overrides = {
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}}
        }

        try:
            info = self._download_with_opts(
                url,
                output_template,
                format_selector=(
                    "bestaudio[acodec!=none]/"
                    "bestaudio/"
                    "b[acodec!=none]"
                ),
                attempt_label="audio_only_default",
                max_height=None,
                ydl_overrides={
                    "noplaylist": True,
                },
            )
        except yt_dlp.utils.DownloadError as exc:
            if not self._is_youtube_url(url):
                raise
            logger.warning(
                "yt-dlp default-client audio download failed for %s; retrying with legacy YouTube clients: %s",
                video_id,
                exc,
            )
            info = self._download_with_opts(
                url,
                output_template,
                format_selector=(
                    "bestaudio[acodec!=none]/"
                    "bestaudio/"
                    "b[acodec!=none]"
                ),
                attempt_label="audio_only_legacy_clients",
                max_height=None,
                ydl_overrides=legacy_client_overrides,
            )

        downloaded_path = self._resolve_output_path(output_template)
        return {
            "path": downloaded_path,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "platform": info.get("extractor_key", "unknown").lower(),
            "external_id": info.get("id"),
        }

    def _extract_with_proxy_rotation(self, url: str, ydl_opts: dict) -> dict:
        """Try extract_info, rotating through proxies on bot-check failures."""
        proxies = _build_proxy_list()
        last_exc: Exception | None = None

        # First attempt uses whatever proxy _base_ydl_opts already set.
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            reason = classify_yt_dlp_error(exc)
            if reason != "youtube_bot_check" or not proxies:
                logger.warning("yt-dlp probe failed (%s) for %s: %s", reason, url, exc)
                raise VideoProbeError(reason, str(exc)) from exc
            logger.warning(
                "yt-dlp bot check with proxy %s; rotating through %d proxies",
                ydl_opts.get("proxy", "none"),
                len(proxies),
            )
            last_exc = exc

        # Rotate through remaining proxies (skip index 0 — already tried).
        for i, proxy in enumerate(proxies[1:], start=2):
            ydl_opts["proxy"] = proxy
            try:
                logger.info("yt-dlp retry with proxy %d/%d", i, len(proxies))
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            except Exception as exc:
                reason = classify_yt_dlp_error(exc)
                if reason != "youtube_bot_check":
                    logger.warning("yt-dlp probe failed (%s) for %s: %s", reason, url, exc)
                    raise VideoProbeError(reason, str(exc)) from exc
                logger.warning("yt-dlp bot check with proxy %d/%d", i, len(proxies))
                last_exc = exc

        reason = classify_yt_dlp_error(last_exc)
        logger.warning(
            "yt-dlp probe failed after all %d proxies for %s: %s",
            len(proxies), url, last_exc,
        )
        raise VideoProbeError(reason, str(last_exc)) from last_exc

    def probe_url(self, url: str) -> dict:
        """Validate URL with yt-dlp and return probe metadata.

        This does not download media. It only extracts metadata needed for:
        - can we access downloadable formats?
        - does the source expose subtitles/captions?
        - what is the duration for cost calculation?
        """
        _validate_url(url)
        ydl_opts = self._base_ydl_opts(max_height=1080)
        # Probing doesn't need cookies — stale/IP-mismatched cookies can
        # trigger YouTube bot checks *before* PO tokens come into play.
        ydl_opts.pop("cookiefile", None)
        ydl_opts.update(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
        )

        info = self._extract_with_proxy_rotation(url, ydl_opts)

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

        audio_path = f"{os.path.splitext(video_path)[0]}.wav"

        (
            ffmpeg.input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16k")
            .overwrite_output()
            .run(quiet=True)
        )

        return audio_path


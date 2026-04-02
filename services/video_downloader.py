import html
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import time
from glob import glob
from pathlib import Path
from urllib.parse import urlparse

import ffmpeg as ffmpeg_lib
import httpx
import yt_dlp
from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from config import (
    MAX_VIDEO_SIZE_MB,
    TEMP_DIR,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_SOURCE_FILE,
    YTDLP_DOWNLOAD_RETRIES,
    YTDLP_EXTRACTOR_RETRIES,
    YTDLP_FRAGMENT_RETRIES,
    YTDLP_PROXY,
    YTDLP_PROXY_LIST,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
)

_MIN_FREE_DISK_MB = int(os.getenv("MIN_FREE_DISK_MB", str(MAX_VIDEO_SIZE_MB * 2 + 500)))


def _build_yt_transcript_api(proxy_url: str | None = None) -> YouTubeTranscriptApi:
    """Build YouTubeTranscriptApi with an optional proxy."""
    proxy_url = str(proxy_url or "").strip() or None
    if proxy_url:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=proxy_url,
                https_url=proxy_url,
            )
        )
    return YouTubeTranscriptApi()


def _build_yt_transcript_api_candidates() -> list[tuple[str | None, YouTubeTranscriptApi]]:
    proxy_urls: list[str | None] = []
    if YTDLP_PROXY:
        proxy_urls.append(YTDLP_PROXY)
    for proxy_url in YTDLP_PROXY_LIST:
        if proxy_url not in proxy_urls:
            proxy_urls.append(proxy_url)
    if not proxy_urls:
        proxy_urls.append(None)
    return [(proxy_url, _build_yt_transcript_api(proxy_url)) for proxy_url in proxy_urls]


def _is_retryable_youtube_transcript_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            VideoUnplayable,
            AgeRestricted,
            InvalidVideoId,
        ),
    ):
        return False
    return isinstance(
        exc,
        (
            RequestBlocked,
            IpBlocked,
            YouTubeRequestFailed,
            CouldNotRetrieveTranscript,
        ),
    )


def _fetch_youtube_transcript_for_client(
    transcript_api: YouTubeTranscriptApi,
    video_id: str,
    preferred_languages: list[str],
):
    try:
        return transcript_api.fetch(video_id, languages=preferred_languages)
    except NoTranscriptFound:
        return transcript_api.fetch(video_id)

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


_YOUTUBE_MEDIA_DOWNLOAD_BLOCK_MESSAGES = (
    "http error 403",
    "unable to download video data",
    "requires a gvs po token",
    "po token",
    "requested format is not available",
)
_LOGGED_COOKIE_SOURCE_WARNINGS: set[str] = set()
_INCOMPLETE_DOWNLOAD_SUFFIXES = {".part", ".ytdl", ".temp", ".tmp"}


_SUPPORTED_SUBTITLE_EXTENSIONS = ("json3", "vtt", "srt")
_SUBTITLE_EXTENSION_PRIORITY = {
    ext: index for index, ext in enumerate(_SUPPORTED_SUBTITLE_EXTENSIONS)
}
_TIMESTAMP_TAG_RE = re.compile(r"<[^>]+>")
_SRT_BLOCK_SPLIT_RE = re.compile(r"\r?\n\r?\n+")
_WEBVTT_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{3})|\d{1,2}:\d{2}(?:[.,]\d{3}))\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{3})|\d{1,2}:\d{2}(?:[.,]\d{3}))"
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


def _normalize_language_code(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return text.replace("_", "-")


def _strip_subtitle_markup(text: str) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = _TIMESTAMP_TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    return " ".join(cleaned.split()).strip()


def _parse_timestamp_to_seconds(value: str) -> float | None:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None

    parts = text.split(":")
    try:
        numeric_parts = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numeric_parts) == 2:
        minutes, seconds = numeric_parts
        return minutes * 60.0 + seconds
    if len(numeric_parts) == 3:
        hours, minutes, seconds = numeric_parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    return None


def _build_transcript_payload(
    *,
    segments: list[dict],
    source: str,
    language: str | None,
) -> dict | None:
    valid_segments = [
        {
            "id": index,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment["text"]).strip(),
        }
        for index, segment in enumerate(segments)
        if float(segment["end"]) > float(segment["start"]) and str(segment["text"]).strip()
    ]
    if not valid_segments:
        return None

    language_code = _normalize_language_code(language)
    return {
        "text": " ".join(segment["text"] for segment in valid_segments).strip(),
        "segments": valid_segments,
        "source": source,
        "language": language_code or "unknown",
        "languageCode": language_code or "unknown",
    }


def _parse_webvtt_payload(payload_text: str) -> list[dict]:
    lines = payload_text.splitlines()
    segments: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if (
            not line
            or line.upper() == "WEBVTT"
            or line.startswith("NOTE")
            or line.startswith("STYLE")
            or line.startswith("REGION")
        ):
            index += 1
            continue

        match = _WEBVTT_TIMESTAMP_RE.search(line)
        if match is None:
            index += 1
            continue

        start = _parse_timestamp_to_seconds(match.group("start"))
        end = _parse_timestamp_to_seconds(match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        text = _strip_subtitle_markup(" ".join(text_lines))
        if start is not None and end is not None and end > start and text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _parse_srt_payload(payload_text: str) -> list[dict]:
    segments: list[dict] = []
    for block in _SRT_BLOCK_SPLIT_RE.split(payload_text.strip()):
        lines = [line.strip("\ufeff").strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timestamp_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in timestamp_line:
            continue
        start_text, end_text = [part.strip() for part in timestamp_line.split("-->", 1)]
        start = _parse_timestamp_to_seconds(start_text)
        end = _parse_timestamp_to_seconds(end_text)
        text_lines = lines[2:] if timestamp_line == lines[1] else lines[1:]
        text = _strip_subtitle_markup(" ".join(text_lines))
        if start is not None and end is not None and end > start and text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _parse_json3_payload(payload_text: str) -> list[dict]:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return []

    segments: list[dict] = []
    for event in payload.get("events") or []:
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs")
        try:
            start = float(start_ms) / 1000.0
            end = start + float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            continue

        text = _strip_subtitle_markup(
            "".join(
                str(seg.get("utf8") or "")
                for seg in (event.get("segs") or [])
            )
        )
        if end > start and text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _language_priority(
    *,
    preferred_languages: list[str],
    candidate_language: str | None,
) -> int:
    normalized = _normalize_language_code(candidate_language)
    if normalized is None:
        return len(preferred_languages) + 20

    base = normalized.split("-", 1)[0]
    for index, preferred in enumerate(preferred_languages):
        if normalized == preferred or base == preferred:
            return index
        if preferred.startswith(f"{base}-") or normalized.startswith(f"{preferred}-"):
            return index + len(preferred_languages)
    return len(preferred_languages) + 20


class VideoProbeError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.failure_reason = reason


class VideoDownloadError(RuntimeError):
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
        self._warn_if_configured_cookie_source_unreadable()

    def _default_runtime_cookiefile(self) -> str:
        return os.path.join(self.temp_dir, "yt-dlp", "cookies.txt")

    @staticmethod
    def _warn_cookie_source_unreadable_once(path: str) -> None:
        normalized = os.path.abspath(path)
        if normalized in _LOGGED_COOKIE_SOURCE_WARNINGS:
            return
        _LOGGED_COOKIE_SOURCE_WARNINGS.add(normalized)
        logger.warning(
            "yt-dlp cookie source is configured but not readable: %s; continuing without cookies",
            path,
        )

    def _warn_if_configured_cookie_source_unreadable(self) -> None:
        source_cookie_path = YTDLP_COOKIES_SOURCE_FILE
        if source_cookie_path and not self._is_readable_cookie_path(source_cookie_path):
            self._warn_cookie_source_unreadable_once(source_cookie_path)

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
                self._warn_cookie_source_unreadable_once(source_cookie_path)
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
                "bv*[vcodec!=none]+ba[acodec!=none]/"
                "bv+ba/"
                "b/"
                "bv*+ba/b"
            )

        height_filter = f"[height<={normalized_max_height}]"
        return (
            f"bv*{height_filter}[vcodec!=none]+ba[acodec!=none]/"
            f"bv*{height_filter}+ba/"
            f"b{height_filter}/"
            "bv*+ba/b"
        )

    @classmethod
    def _compatibility_selector_for_max_height(cls, max_height: int | None) -> str:
        normalized_max_height = cls._normalize_max_height(max_height)
        if normalized_max_height is None:
            return (
                "bv[vcodec^=avc1]+ba[acodec^=mp4a]/"
                "bv[vcodec^=avc1]+ba/"
                "b[ext=mp4][vcodec!=none][acodec!=none]/"
                "b[vcodec!=none][acodec!=none]"
            )

        height_filter = f"[height<={normalized_max_height}]"
        return (
            f"bv{height_filter}[vcodec^=avc1]+ba[acodec^=mp4a]/"
            f"bv{height_filter}[vcodec^=avc1]+ba/"
            f"b{height_filter}[ext=mp4][vcodec!=none][acodec!=none]/"
            f"b{height_filter}[vcodec!=none][acodec!=none]"
        )

    def _base_ydl_opts(self, *, max_height: int | None = 1080) -> dict:
        opts: dict = {
            # Prefer the highest-quality source first and fall back to
            # compatibility-oriented selectors only if needed.
            "format": self._format_selector_for_max_height(max_height),
            "merge_output_format": "mkv",
            "noplaylist": True,
            # Let yt-dlp use its built-in default User-Agent (keeps Chrome
            # version current and avoids stale-UA bot detection).
            "js_runtimes": {"node": {}},
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
    def _probe_stream_types(media_path: str) -> set[str]:
        try:
            probe = ffmpeg_lib.probe(media_path)
        except Exception as exc:
            logger.warning("Failed to probe media streams in %s: %s", media_path, exc)
            return set()
        return {
            codec_type
            for stream in probe.get("streams", [])
            if (codec_type := str(stream.get("codec_type") or "").strip())
        }

    @staticmethod
    def _has_audio_stream(video_path: str) -> bool:
        return "audio" in VideoDownloader._probe_stream_types(video_path)

    @staticmethod
    def _has_video_stream(video_path: str) -> bool:
        return "video" in VideoDownloader._probe_stream_types(video_path)

    @staticmethod
    def _has_muxed_audio_video(video_path: str) -> bool:
        stream_types = VideoDownloader._probe_stream_types(video_path)
        return "video" in stream_types and "audio" in stream_types

    @staticmethod
    def _is_incomplete_download_artifact(path: str) -> bool:
        suffixes = {suffix.lower() for suffix in Path(str(path)).suffixes}
        return bool(suffixes & _INCOMPLETE_DOWNLOAD_SUFFIXES)

    @staticmethod
    def _resolve_output_path(expected_path: str, *, require_video: bool = False) -> str:
        base, _ = os.path.splitext(expected_path)
        candidate_pool: list[str] = []
        if os.path.isfile(expected_path) and not VideoDownloader._is_incomplete_download_artifact(
            expected_path
        ):
            candidate_pool.append(expected_path)
        candidate_pool.extend(
            path
            for path in glob(f"{base}.*")
            if (
                os.path.isfile(path)
                and path not in candidate_pool
                and not VideoDownloader._is_incomplete_download_artifact(path)
            )
        )
        candidates = candidate_pool
        if not candidates:
            return expected_path

        muxed_candidates = [
            path for path in candidates if VideoDownloader._has_muxed_audio_video(path)
        ]
        if muxed_candidates:
            return max(muxed_candidates, key=os.path.getmtime)

        video_candidates = [path for path in candidates if VideoDownloader._has_video_stream(path)]
        if video_candidates:
            return max(video_candidates, key=os.path.getmtime)
        if require_video:
            return expected_path
        return max(candidates, key=os.path.getmtime)

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        hostname = (urlparse(url).hostname or "").strip().lower()
        return hostname in _YOUTUBE_DOMAINS

    @staticmethod
    def _has_cookie_configuration() -> bool:
        return bool(YTDLP_COOKIES_SOURCE_FILE) or VideoDownloader._is_readable_cookie_path(
            YTDLP_COOKIES_FILE
        )

    @staticmethod
    def _has_proxy_configuration() -> bool:
        return bool(_build_proxy_list())

    @staticmethod
    def _youtube_direct_public_overrides() -> dict[str, object]:
        return {
            "cookiefile": None,
            "proxy": None,
        }

    @staticmethod
    def _youtube_cookie_first_overrides() -> dict[str, object]:
        return {
            "proxy": None,
        }

    @staticmethod
    def _youtube_proxy_fallback_overrides() -> dict[str, object]:
        return {
            "cookiefile": None,
        }

    @staticmethod
    def _merge_ydl_overrides(*overrides: dict[str, object] | None) -> dict[str, object] | None:
        merged: dict[str, object] = {}
        for override in overrides:
            if not override:
                continue
            for key, value in override.items():
                if (
                    key == "extractor_args"
                    and isinstance(value, dict)
                    and isinstance(merged.get(key), dict)
                ):
                    merged_extractor_args = dict(merged[key])
                    for extractor_name, extractor_value in value.items():
                        if (
                            isinstance(extractor_value, dict)
                            and isinstance(merged_extractor_args.get(extractor_name), dict)
                        ):
                            merged_extractor_args[extractor_name] = {
                                **merged_extractor_args[extractor_name],
                                **extractor_value,
                            }
                        else:
                            merged_extractor_args[extractor_name] = extractor_value
                    merged[key] = merged_extractor_args
                else:
                    merged[key] = value
        return merged or None

    def _probe_ydl_opts(self, *, include_cookies: bool, include_proxy: bool) -> dict:
        opts = self._base_ydl_opts(max_height=2160)
        if not include_cookies:
            opts.pop("cookiefile", None)
        if not include_proxy:
            opts.pop("proxy", None)
        opts.update(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
        )
        return opts

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
        if ydl_opts.get("cookiefile") is None:
            ydl_opts.pop("cookiefile", None)
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
        last_exc: Exception | None = None
        explicit_proxy_override = bool(ydl_overrides and "proxy" in ydl_overrides)
        if explicit_proxy_override:
            attempts = [ydl_opts.get("proxy")]
        else:
            proxies = _build_proxy_list()
            # Try with current proxy, then rotate on bot-check failures.
            attempts = [ydl_opts.get("proxy")] + [
                p for p in proxies if p != ydl_opts.get("proxy")
            ]
        for attempt_idx, proxy in enumerate(attempts):
            if proxy:
                ydl_opts["proxy"] = proxy
            else:
                ydl_opts.pop("proxy", None)
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
    def _friendly_youtube_download_error(exc: Exception) -> VideoDownloadError:
        message = str(exc).strip()
        normalized_message = message.lower()
        reason = classify_yt_dlp_error(exc)
        if (
            reason in {"youtube_bot_check", "youtube_auth_required"}
            or any(token in normalized_message for token in _YOUTUBE_MEDIA_DOWNLOAD_BLOCK_MESSAGES)
        ):
            return VideoDownloadError(
                "youtube_media_download_blocked",
                "YouTube media download blocked; refresh mounted cookies and retry. "
                f"yt-dlp reported: {message}",
            )
        return VideoDownloadError(
            "youtube_download_failed",
            f"YouTube media download failed: {message}",
        )

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

    def _download_youtube_with_fallbacks(
        self,
        url: str,
        output_path: str,
        *,
        format_selector: str,
        attempt_label: str,
        proxy_fallback_label: str,
        direct_public_label: str,
        legacy_label: str,
        video_id: str,
        max_height: int | None = 1080,
        ydl_overrides: dict[str, object] | None = None,
    ) -> dict:
        legacy_client_overrides = {
            "extractor_args": {
                "youtube": {
                    "player_client": ["tv_downgraded", "web_safari", "web"]
                }
            }
        }
        cookie_first_allowed = self._has_cookie_configuration()
        proxy_allowed = self._has_proxy_configuration()
        initial_overrides = ydl_overrides
        if cookie_first_allowed:
            initial_overrides = self._merge_ydl_overrides(
                ydl_overrides,
                self._youtube_cookie_first_overrides(),
            )

        try:
            return self._download_with_opts(
                url,
                output_path,
                format_selector=format_selector,
                attempt_label=attempt_label,
                max_height=max_height,
                ydl_overrides=initial_overrides,
            )
        except yt_dlp.utils.DownloadError as exc:
            last_attempt_label = attempt_label
            last_exc: yt_dlp.utils.DownloadError = exc

            if cookie_first_allowed and proxy_allowed:
                logger.warning(
                    "yt-dlp download failed (%s) for %s; retrying with configured proxies: %s",
                    attempt_label,
                    video_id,
                    exc,
                )
                try:
                    return self._download_with_opts(
                        url,
                        output_path,
                        format_selector=format_selector,
                        attempt_label=proxy_fallback_label,
                        max_height=max_height,
                        ydl_overrides=self._merge_ydl_overrides(
                            ydl_overrides,
                            self._youtube_proxy_fallback_overrides(),
                        ),
                    )
                except yt_dlp.utils.DownloadError as proxy_exc:
                    last_attempt_label = proxy_fallback_label
                    last_exc = proxy_exc

            direct_public_allowed = cookie_first_allowed or proxy_allowed
            if direct_public_allowed:
                logger.warning(
                    "yt-dlp download failed (%s) for %s; retrying with direct public YouTube access: %s",
                    last_attempt_label,
                    video_id,
                    last_exc,
                )
                try:
                    return self._download_with_opts(
                        url,
                        output_path,
                        format_selector=format_selector,
                        attempt_label=direct_public_label,
                        max_height=max_height,
                        ydl_overrides=self._merge_ydl_overrides(
                            ydl_overrides,
                            self._youtube_direct_public_overrides(),
                        ),
                    )
                except yt_dlp.utils.DownloadError as direct_exc:
                    logger.warning(
                        "yt-dlp direct-public YouTube download failed for %s; retrying with legacy YouTube clients: %s",
                        video_id,
                        direct_exc,
                    )
            else:
                logger.warning(
                    "yt-dlp download failed (%s) for %s; retrying with legacy YouTube clients: %s",
                    last_attempt_label,
                    video_id,
                    last_exc,
                )
            try:
                return self._download_with_opts(
                    url,
                    output_path,
                    format_selector=format_selector,
                    attempt_label=legacy_label,
                    max_height=max_height,
                    ydl_overrides=self._merge_ydl_overrides(
                        ydl_overrides,
                        legacy_client_overrides,
                    ),
                )
            except yt_dlp.utils.DownloadError as final_exc:
                raise self._friendly_youtube_download_error(final_exc) from final_exc

    def download(self, url: str, video_id: str, *, max_height: int | None = 1080) -> dict:
        """Download video and return metadata."""
        _validate_url(url)
        self._check_disk_space(self.temp_dir)
        output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")
        normalized_max_height = self._normalize_max_height(max_height)
        primary_selector = self._format_selector_for_max_height(normalized_max_height)
        compatibility_selector = self._compatibility_selector_for_max_height(
            normalized_max_height
        )
        if self._is_youtube_url(url):
            info = self._download_youtube_with_fallbacks(
                url,
                output_path,
                format_selector=primary_selector,
                attempt_label="best_source_default",
                proxy_fallback_label="best_source_proxy_fallback",
                direct_public_label="best_source_direct_public",
                legacy_label="best_source_legacy_clients",
                video_id=video_id,
                max_height=normalized_max_height,
            )
        else:
            info = self._download_with_opts(
                url,
                output_path,
                format_selector=primary_selector,
                attempt_label="best_source_default",
                max_height=normalized_max_height,
            )
        downloaded_path = self._resolve_output_path(output_path, require_video=True)

        if not self._has_audio_stream(downloaded_path):
            logger.warning(
                "Downloaded media for %s has no audio stream; retrying with compatibility selector",
                video_id,
            )
            try:
                os.remove(downloaded_path)
            except OSError:
                pass

            try:
                if self._is_youtube_url(url):
                    info = self._download_youtube_with_fallbacks(
                        url,
                        output_path,
                        format_selector=compatibility_selector,
                        attempt_label="compatibility_fallback",
                        proxy_fallback_label="compatibility_proxy_fallback",
                        direct_public_label="compatibility_direct_public",
                        legacy_label="compatibility_legacy_clients",
                        video_id=video_id,
                        max_height=normalized_max_height,
                    )
                else:
                    info = self._download_with_opts(
                        url,
                        output_path,
                        format_selector=compatibility_selector,
                        attempt_label="compatibility_fallback",
                        max_height=normalized_max_height,
                    )
            except yt_dlp.utils.DownloadError as exc:
                if self._is_youtube_url(url):
                    raise self._friendly_youtube_download_error(exc) from exc
                raise
            downloaded_path = self._resolve_output_path(output_path, require_video=True)

        if not self._has_audio_stream(downloaded_path):
            logger.warning("Downloaded media for %s still has no audio stream", video_id)

        return {
            "path": downloaded_path,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "platform": info.get("extractor_key", "unknown").lower(),
            "external_id": info.get("id"),
            "selected_formats": self._selected_format_summary(info),
        }

    def download_audio_only(self, url: str, video_id: str) -> dict:
        """Download source audio without fetching the full video stream."""
        _validate_url(url)
        output_template = os.path.join(self.temp_dir, f"{video_id}.audio.%(ext)s")
        audio_selector = (
            "bestaudio[acodec!=none]/"
            "bestaudio/"
            "b[acodec!=none]"
        )
        if self._is_youtube_url(url):
            info = self._download_youtube_with_fallbacks(
                url,
                output_template,
                format_selector=audio_selector,
                attempt_label="audio_only_default",
                proxy_fallback_label="audio_only_proxy_fallback",
                direct_public_label="audio_only_direct_public",
                legacy_label="audio_only_legacy_clients",
                video_id=video_id,
                max_height=None,
                ydl_overrides={
                    "noplaylist": True,
                },
            )
        else:
            info = self._download_with_opts(
                url,
                output_template,
                format_selector=audio_selector,
                attempt_label="audio_only_default",
                max_height=None,
                ydl_overrides={
                    "noplaylist": True,
                },
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

    def _extract_with_proxy_rotation(
        self,
        url: str,
        ydl_opts: dict,
        *,
        rotate_on_bot_check: bool = True,
    ) -> dict:
        """Try extract_info, rotating through proxies on bot-check failures."""
        proxies = _build_proxy_list()
        last_exc: Exception | None = None

        # First attempt uses whatever proxy _base_ydl_opts already set.
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            reason = classify_yt_dlp_error(exc)
            if reason != "youtube_bot_check" or not proxies or not rotate_on_bot_check:
                logger.warning("yt-dlp probe failed (%s) for %s: %s", reason, url, exc)
                raise VideoProbeError(reason, str(exc)) from exc
            logger.warning(
                "yt-dlp bot check with proxy %s; rotating through %d proxies",
                ydl_opts.get("proxy", "none"),
                len(proxies),
            )
            last_exc = exc

        # Rotate through remaining proxies (skip index 0 — already tried).
        initial_proxy = ydl_opts.get("proxy")
        start_index = 0
        if isinstance(initial_proxy, str) and initial_proxy in proxies:
            start_index = proxies.index(initial_proxy) + 1

        for i, proxy in enumerate(proxies[start_index:], start=start_index + 1):
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

    def _probe_info(self, url: str) -> dict:
        """Fetch source metadata from yt-dlp without downloading media."""
        _validate_url(url)
        cookie_probe_opts = self._probe_ydl_opts(include_cookies=True, include_proxy=False)
        cookiefile = cookie_probe_opts.get("cookiefile")
        # Probing doesn't need cookies — stale/IP-mismatched cookies can
        # trigger YouTube bot checks *before* PO tokens come into play.
        if cookiefile:
            try:
                return self._extract_with_proxy_rotation(
                    url,
                    cookie_probe_opts,
                    rotate_on_bot_check=False,
                )
            except VideoProbeError as exc:
                logger.warning(
                    "yt-dlp probe failed with cookies (%s) for %s; retrying with configured proxies",
                    exc.failure_reason,
                    url,
                )

        proxy_probe_opts = self._probe_ydl_opts(include_cookies=False, include_proxy=True)
        return self._extract_with_proxy_rotation(url, proxy_probe_opts)

    def _preferred_languages(self, language_hint: str | None = None) -> list[str]:
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

    def _pick_provider_subtitle_track(
        self,
        info: dict,
        *,
        preferred_languages: list[str],
    ) -> dict[str, object]:
        selected_track: dict[str, object] | None = None
        selected_rank: tuple[int, int, int] | None = None
        unsupported_languages: set[str] = set()
        empty_languages: set[str] = set()

        for kind, track_map in (
            ("provider_subtitles", info.get("subtitles") or {}),
            ("provider_automatic_captions", info.get("automatic_captions") or {}),
        ):
            if not isinstance(track_map, dict):
                continue
            source_rank = 0 if kind == "provider_subtitles" else 1
            for language, entries in track_map.items():
                normalized_language = _normalize_language_code(language) or "unknown"
                if not isinstance(entries, list) or not entries:
                    empty_languages.add(normalized_language)
                    continue

                supported_entries = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    ext = str(entry.get("ext") or "").strip().lower()
                    url = str(entry.get("url") or "").strip()
                    if not url:
                        continue
                    if ext not in _SUBTITLE_EXTENSION_PRIORITY:
                        continue
                    supported_entries.append(
                        {
                            "kind": kind,
                            "language": normalized_language,
                            "ext": ext,
                            "url": url,
                        }
                    )

                if not supported_entries:
                    unsupported_languages.add(normalized_language)
                    continue

                supported_entries.sort(
                    key=lambda entry: _SUBTITLE_EXTENSION_PRIORITY[entry["ext"]]
                )
                candidate = supported_entries[0]
                rank = (
                    source_rank,
                    _language_priority(
                        preferred_languages=preferred_languages,
                        candidate_language=str(candidate["language"]),
                    ),
                    _SUBTITLE_EXTENSION_PRIORITY[str(candidate["ext"])],
                )
                if selected_rank is None or rank < selected_rank:
                    selected_track = candidate
                    selected_rank = rank

        fallback_reason: str | None = None
        if selected_track is None:
            if unsupported_languages:
                fallback_reason = (
                    "provider_caption_unsupported_format:"
                    + ",".join(sorted(unsupported_languages))
                )
            elif empty_languages:
                fallback_reason = (
                    "provider_caption_missing_url:" + ",".join(sorted(empty_languages))
                )
            else:
                fallback_reason = "provider_caption_unavailable"

        return {
            "track": selected_track,
            "fallback_reason": fallback_reason,
        }

    def _parse_provider_subtitle_payload(
        self,
        *,
        subtitle_url: str,
        subtitle_ext: str,
        language: str | None,
    ) -> dict | None:
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(subtitle_url)
                response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "Failed to fetch provider subtitle payload (%s): %s",
                subtitle_url,
                exc,
            )
            return None

        payload_text = response.text
        subtitle_ext = str(subtitle_ext or "").strip().lower()
        if subtitle_ext == "vtt":
            segments = _parse_webvtt_payload(payload_text)
        elif subtitle_ext == "srt":
            segments = _parse_srt_payload(payload_text)
        elif subtitle_ext == "json3":
            segments = _parse_json3_payload(payload_text)
        else:
            return None

        return _build_transcript_payload(
            segments=segments,
            source="provider_captions",
            language=language,
        )

    def probe_url(self, url: str) -> dict:
        """Validate URL with yt-dlp and return probe metadata.

        This does not download media. It only extracts metadata needed for:
        - can we access downloadable formats?
        - does the source expose subtitles/captions?
        - what is the duration for cost calculation?
        """
        info = self._probe_info(url)

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

    def get_provider_transcript(
        self,
        url: str,
        *,
        preferred_languages: list[str] | None = None,
    ) -> dict[str, object]:
        info = self._probe_info(url)
        preferred = preferred_languages or self._preferred_languages()
        selection = self._pick_provider_subtitle_track(
            info,
            preferred_languages=preferred,
        )
        track = selection.get("track")
        if not isinstance(track, dict):
            return {
                "transcript": None,
                "fallback_reason": selection.get("fallback_reason"),
                "track_source": None,
                "track_ext": None,
                "track_language": None,
            }

        transcript = self._parse_provider_subtitle_payload(
            subtitle_url=str(track["url"]),
            subtitle_ext=str(track["ext"]),
            language=str(track["language"]),
        )
        if transcript is None:
            return {
                "transcript": None,
                "fallback_reason": f"provider_caption_parse_failed:{track['ext']}",
                "track_source": str(track["kind"]),
                "track_ext": str(track["ext"]),
                "track_language": str(track["language"]),
            }

        return {
            "transcript": transcript,
            "fallback_reason": None,
            "track_source": str(track["kind"]),
            "track_ext": str(track["ext"]),
            "track_language": str(track["language"]),
        }

    def get_youtube_transcript(
        self,
        video_id: str,
        *,
        preferred_languages: list[str] | None = None,
    ) -> dict | None:
        """Get transcript directly from YouTube (v1.x API).

        Tries preferred languages first, then falls back to any available language.
        """
        fetched = None
        preferred = preferred_languages or self._preferred_languages()
        transcript_clients = _build_yt_transcript_api_candidates()

        for attempt_index, (_proxy_url, transcript_api) in enumerate(
            transcript_clients,
            start=1,
        ):
            try:
                fetched = _fetch_youtube_transcript_for_client(
                    transcript_api,
                    video_id,
                    preferred,
                )
                logger.info(
                    "Found %s transcript (%s) for %s",
                    fetched.language,
                    fetched.language_code,
                    video_id,
                )
                break
            except Exception as exc:
                if (
                    attempt_index < len(transcript_clients)
                    and _is_retryable_youtube_transcript_error(exc)
                ):
                    logger.warning(
                        "YouTube transcript fetch blocked on proxy %d/%d for %s; rotating",
                        attempt_index,
                        len(transcript_clients),
                        video_id,
                    )
                    continue
                logger.warning("Could not get YouTube transcript for %s: %s", video_id, exc)
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


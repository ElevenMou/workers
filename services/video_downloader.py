import logging
import os

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from config import TEMP_DIR, MAX_VIDEO_SIZE_MB

_yt_transcript_api = YouTubeTranscriptApi()

logger = logging.getLogger(__name__)


class VideoDownloader:
    def __init__(self, work_dir: str | None = None):
        self.temp_dir = work_dir or TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def download(self, url: str, video_id: str) -> dict:
        """Download video and return metadata."""
        output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")

        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
            "outtmpl": output_path,
            "quiet": False,
            "no_warnings": False,
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
            # Try multiple YouTube client profiles to avoid signature/403 issues
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
            # Set a desktop UA to reduce throttling/403
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            },
            "retries": 3,
            "fragment_retries": 3,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            return {
                "path": output_path,
                "title": info.get("title"),
                "duration": info.get("duration"),
                "thumbnail": info.get("thumbnail"),
                "platform": info.get("extractor_key", "unknown").lower(),
                "external_id": info.get("id"),
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

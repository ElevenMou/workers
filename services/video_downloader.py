import yt_dlp
import os
from youtube_transcript_api import YouTubeTranscriptApi
from config import TEMP_DIR, MAX_VIDEO_SIZE_MB


class VideoDownloader:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def download(self, url: str, video_id: str) -> dict:
        """Download video and return metadata"""
        output_path = os.path.join(self.temp_dir, f"{video_id}.mp4")

        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
            "outtmpl": output_path,
            "quiet": False,
            "no_warnings": False,
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
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

    def get_youtube_transcript(self, video_id: str) -> dict:
        """Get transcript directly from YouTube"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Convert to our format
            segments = []
            for i, entry in enumerate(transcript_list):
                segments.append(
                    {
                        "id": i,
                        "start": entry["start"],
                        "end": entry["start"] + entry["duration"],
                        "text": entry["text"],
                    }
                )

            full_text = " ".join([s["text"] for s in segments])

            return {"text": full_text, "segments": segments, "source": "youtube"}
        except Exception as e:
            print(f"Could not get YouTube transcript: {e}")
            return None

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video"""
        import ffmpeg

        audio_path = video_path.replace(".mp4", ".wav")

        (
            ffmpeg.input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16k")
            .overwrite_output()
            .run(quiet=True)
        )

        return audio_path

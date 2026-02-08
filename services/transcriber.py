import whisper
from config import WHISPER_MODEL


class Transcriber:
    def __init__(self):
        self.model = whisper.load_model(WHISPER_MODEL)

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio with word-level timestamps"""
        result = self.model.transcribe(audio_path, word_timestamps=True, language="en")

        return {"text": result["text"], "segments": result["segments"]}

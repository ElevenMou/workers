import logging
import threading

import whisper
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model singleton - loaded once per process, reused across all jobs.
# Avoids the ~30 s+ reload (and multi-GB memory churn) on every task call.
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-check after acquiring lock
                logger.info("Loading Whisper model '%s' ...", WHISPER_MODEL)
                _model = whisper.load_model(WHISPER_MODEL)
                logger.info("Whisper model loaded.")
    return _model


class Transcriber:
    def __init__(self):
        self.model = _get_model()

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio with word-level timestamps."""
        logger.info("Transcribing %s", audio_path)
        result = self.model.transcribe(audio_path, word_timestamps=True, language="en")
        language = str(result.get("language") or "en").strip().lower()
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": language,
            "languageCode": language,
        }

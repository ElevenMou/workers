import logging
import threading

import whisper
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model singleton(s) - loaded once per process, reused across all jobs.
# Models are keyed by model name so clip generation can use a faster model
# without reloading the default analysis model every task.
# ---------------------------------------------------------------------------
_models: dict[str, object] = {}
_model_lock = threading.Lock()


def _resolve_model_name(model_name: str | None = None) -> str:
    normalized = str(model_name or WHISPER_MODEL).strip()
    return normalized or WHISPER_MODEL


def _get_model(model_name: str | None = None):
    resolved_name = _resolve_model_name(model_name)
    if resolved_name in _models:
        return _models[resolved_name]

    with _model_lock:
        if resolved_name in _models:
            return _models[resolved_name]

        logger.info("Loading Whisper model '%s' ...", resolved_name)
        _models[resolved_name] = whisper.load_model(resolved_name)
        logger.info("Whisper model '%s' loaded.", resolved_name)
    return _models[resolved_name]


class Transcriber:
    def __init__(self, model_name: str | None = None):
        self.model_name = _resolve_model_name(model_name)
        self.model = _get_model(self.model_name)

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio with word-level timestamps."""
        logger.info("Transcribing %s (Whisper model '%s')", audio_path, self.model_name)
        result = self.model.transcribe(audio_path, word_timestamps=True, language="en")
        language = str(result.get("language") or "en").strip().lower()
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": language,
            "languageCode": language,
        }

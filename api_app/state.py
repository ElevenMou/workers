"""API process state and shared logger."""

import logging

from utils.logging_config import setup_logging

setup_logging(component="api")
logger = logging.getLogger("clipry.api")

_WHISPER_READY: bool | None = None


def whisper_ready() -> bool:
    """Lazily verify Whisper is usable on this server process."""
    global _WHISPER_READY
    if _WHISPER_READY is not None:
        return _WHISPER_READY

    try:
        from services.transcriber import Transcriber

        Transcriber()
        _WHISPER_READY = True
    except Exception as exc:
        logger.warning("Whisper readiness check failed: %s", exc)
        _WHISPER_READY = False

    return _WHISPER_READY

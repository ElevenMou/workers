"""API process state and shared logger."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("clipry.api")

_WHISPER_READY: bool | None = None


def whisper_ready() -> bool:
    """Lazily verify Whisper is usable on this server process."""
    global _WHISPER_READY
    if _WHISPER_READY is not None:
        return _WHISPER_READY

    try:
        # Lazy import to avoid loading Whisper unless this check is needed.
        from services.transcriber import Transcriber

        Transcriber()
        _WHISPER_READY = True
    except Exception as exc:
        logger.warning("Whisper readiness check failed: %s", exc)
        _WHISPER_READY = False

    return _WHISPER_READY

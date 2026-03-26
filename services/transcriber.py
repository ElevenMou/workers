import importlib
import logging
import os
import threading
import time

from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model singleton(s) - loaded once per process, reused across all jobs.
# Models are keyed by model name so clip generation can use a faster model
# without reloading the default analysis model every task.
# ---------------------------------------------------------------------------
_models: dict[str, object] = {}
_model_lock = threading.Lock()
_MODEL_LOCK_TIMEOUT_SECONDS = max(
    30.0,
    float(os.getenv("WHISPER_MODEL_LOCK_TIMEOUT_SECONDS", "600")),
)
_MODEL_LOCK_POLL_SECONDS = max(
    0.05,
    float(os.getenv("WHISPER_MODEL_LOCK_POLL_SECONDS", "0.2")),
)
_MODEL_LOCK_STALE_SECONDS = max(
    _MODEL_LOCK_TIMEOUT_SECONDS,
    float(os.getenv("WHISPER_MODEL_LOCK_STALE_SECONDS", "900")),
)
_CHECKSUM_MISMATCH_MARKERS = (
    "sha256 checksum does not not match",
    "sha256 checksum does not match",
)
_whisper_module = None


def _get_whisper_module():
    global _whisper_module
    if _whisper_module is not None:
        return _whisper_module

    try:
        _whisper_module = importlib.import_module("whisper")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Whisper is not installed in this environment.") from exc

    return _whisper_module


def _resolve_model_name(model_name: str | None = None) -> str:
    normalized = str(model_name or WHISPER_MODEL).strip()
    return normalized or WHISPER_MODEL


def _whisper_cache_dir() -> str:
    configured = str(os.getenv("WHISPER_CACHE_DIR", "")).strip()
    if configured:
        return configured
    xdg_cache_home = str(os.getenv("XDG_CACHE_HOME", "")).strip()
    base_dir = xdg_cache_home or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base_dir, "whisper")


def _whisper_model_cache_path(model_name: str) -> str | None:
    whisper_module = _get_whisper_module()
    model_url = getattr(whisper_module, "_MODELS", {}).get(model_name)
    if not model_url:
        return None
    return os.path.join(_whisper_cache_dir(), os.path.basename(str(model_url)))


def _is_checksum_mismatch_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return any(marker in text for marker in _CHECKSUM_MISMATCH_MARKERS)


def _delete_corrupt_model_cache(model_name: str) -> None:
    cache_path = _whisper_model_cache_path(model_name)
    if not cache_path:
        return
    try:
        os.remove(cache_path)
        logger.warning(
            "Removed corrupt Whisper cache file for '%s': %s",
            model_name,
            cache_path,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            "Failed to remove corrupt Whisper cache file for '%s' at %s: %s",
            model_name,
            cache_path,
            exc,
        )


class _InterProcessFileLock:
    def __init__(
        self,
        path: str,
        *,
        timeout_seconds: float,
        poll_seconds: float,
        stale_seconds: float,
    ):
        self.path = path
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.stale_seconds = max(self.timeout_seconds, float(stale_seconds))
        self._fd: int | None = None

    def _remove_if_stale(self) -> bool:
        try:
            stat_result = os.stat(self.path)
        except FileNotFoundError:
            return False

        age_seconds = max(0.0, time.time() - stat_result.st_mtime)
        if age_seconds <= self.stale_seconds:
            return False

        try:
            os.remove(self.path)
            logger.warning(
                "Removed stale Whisper model lock file %s (age=%.1fs)",
                self.path,
                age_seconds,
            )
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def __enter__(self):
        lock_dir = os.path.dirname(self.path) or "."
        os.makedirs(lock_dir, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds

        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = f"{os.getpid()}\n{time.time():.6f}\n".encode("utf-8")
                os.write(self._fd, payload)
                return self
            except FileExistsError:
                if self._remove_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for Whisper model lock: {self.path}"
                    )
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError as lock_exc:
            logger.warning("Failed to remove Whisper model lock %s: %s", self.path, lock_exc)
        return False


def _load_model_with_repair(model_name: str):
    whisper_module = _get_whisper_module()
    download_root = _whisper_cache_dir()
    os.makedirs(download_root, exist_ok=True)
    for attempt in range(2):
        try:
            return whisper_module.load_model(model_name, download_root=download_root)
        except RuntimeError as exc:
            if attempt >= 1 or not _is_checksum_mismatch_error(exc):
                raise
            _delete_corrupt_model_cache(model_name)
            logger.warning(
                "Retrying Whisper model '%s' load after checksum mismatch",
                model_name,
            )


def _get_model(model_name: str | None = None):
    resolved_name = _resolve_model_name(model_name)
    if resolved_name in _models:
        return _models[resolved_name]

    with _model_lock:
        if resolved_name in _models:
            return _models[resolved_name]

        cache_path = _whisper_model_cache_path(resolved_name)
        lock_path = f"{cache_path}.lock" if cache_path else os.path.join(
            _whisper_cache_dir(),
            f"{resolved_name.replace(os.sep, '_')}.lock",
        )
        with _InterProcessFileLock(
            lock_path,
            timeout_seconds=_MODEL_LOCK_TIMEOUT_SECONDS,
            poll_seconds=_MODEL_LOCK_POLL_SECONDS,
            stale_seconds=_MODEL_LOCK_STALE_SECONDS,
        ):
            if resolved_name in _models:
                return _models[resolved_name]

            logger.info("Loading Whisper model '%s' ...", resolved_name)
            _models[resolved_name] = _load_model_with_repair(resolved_name)
        logger.info("Whisper model '%s' loaded.", resolved_name)
    return _models[resolved_name]


class Transcriber:
    def __init__(self, model_name: str | None = None):
        self.model_name = _resolve_model_name(model_name)
        self.model = _get_model(self.model_name)

    @staticmethod
    def _normalize_language_hint(language_hint: str | None) -> str | None:
        text = str(language_hint or "").strip().lower()
        if not text:
            return None
        base = text.split("-", 1)[0].split("_", 1)[0].strip()
        return base or None

    def transcribe(
        self,
        audio_path: str,
        *,
        language_hint: str | None = None,
        word_timestamps: bool = True,
    ) -> dict:
        """Transcribe audio with word-level timestamps."""
        logger.info("Transcribing %s (Whisper model '%s')", audio_path, self.model_name)
        transcribe_kwargs: dict[str, object] = {}
        if word_timestamps:
            transcribe_kwargs["word_timestamps"] = True
        normalized_hint = self._normalize_language_hint(language_hint)
        if normalized_hint:
            transcribe_kwargs["language"] = normalized_hint
        result = self.model.transcribe(audio_path, **transcribe_kwargs)
        language = str(result.get("language") or "en").strip().lower()
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": language,
            "languageCode": language,
        }

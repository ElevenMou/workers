import os
import logging
from math import ceil
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        parsed = int(default)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %d", name, value, default)
            parsed = int(default)

    if minimum is not None:
        return max(int(minimum), parsed)
    return parsed


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        parsed = float(default)
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %s", name, value, default)
            parsed = float(default)

    if minimum is not None:
        parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_MAX_CONNECTIONS = _env_int("REDIS_MAX_CONNECTIONS", 20, minimum=5)
REDIS_CONNECTION_ALERT_THRESHOLD = _env_float(
    "REDIS_CONNECTION_ALERT_THRESHOLD",
    0.80,
    minimum=0.05,
    maximum=1.0,
)

# Queue backpressure - reject enqueues above these depths.
MAX_VIDEO_QUEUE_DEPTH = _env_int("MAX_VIDEO_QUEUE_DEPTH", 500, minimum=10)
MAX_CLIP_QUEUE_DEPTH = _env_int("MAX_CLIP_QUEUE_DEPTH", 1000, minimum=10)
MAX_SOCIAL_QUEUE_DEPTH = _env_int("MAX_SOCIAL_QUEUE_DEPTH", 250, minimum=10)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/video_clipper")
RAW_VIDEO_CACHE_DIR = os.getenv(
    "RAW_VIDEO_CACHE_DIR", f"{TEMP_DIR.rstrip('/')}/cache/raw"
)
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 500))

# ---------------------------------------------------------------------------
# Workers - concurrency & timeouts
# ---------------------------------------------------------------------------
NUM_VIDEO_WORKERS = int(os.getenv("NUM_VIDEO_WORKERS", 2))
NUM_CLIP_WORKERS = int(os.getenv("NUM_CLIP_WORKERS", 2))
NUM_SOCIAL_WORKERS = int(os.getenv("NUM_SOCIAL_WORKERS", 1))
VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", 1800))  # 30 min
CLIP_JOB_TIMEOUT = int(os.getenv("CLIP_JOB_TIMEOUT", 1800))  # 30 min
SOCIAL_JOB_TIMEOUT = int(os.getenv("SOCIAL_JOB_TIMEOUT", 1800))  # 30 min
RAW_VIDEO_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("RAW_VIDEO_CLEANUP_INTERVAL_SECONDS", 300)
)
CLIP_ASSET_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("CLIP_ASSET_CLEANUP_INTERVAL_SECONDS", 300)
)
PROCESSING_JOB_STALE_SECONDS = _env_int(
    "PROCESSING_JOB_STALE_SECONDS",
    max(VIDEO_JOB_TIMEOUT, CLIP_JOB_TIMEOUT) + 300,
    minimum=60,
)
SUPERVISOR_ROLE = (
    str(os.getenv("SUPERVISOR_ROLE", "worker")).strip().lower() or "worker"
)
WORKER_INSTANCE_ID = (os.getenv("WORKER_INSTANCE_ID") or "").strip()
MAINTENANCE_LEADER_LOCK_TTL_SECONDS = _env_int(
    "MAINTENANCE_LEADER_LOCK_TTL_SECONDS",
    15,
    minimum=5,
)
MAINTENANCE_LEADER_RENEW_SECONDS = _env_int(
    "MAINTENANCE_LEADER_RENEW_SECONDS",
    5,
    minimum=1,
)
SOCIAL_PUBLICATION_DISPATCH_INTERVAL_SECONDS = _env_int(
    "SOCIAL_PUBLICATION_DISPATCH_INTERVAL_SECONDS",
    30,
    minimum=5,
)
SOCIAL_PUBLICATION_CLAIM_BATCH_SIZE = _env_int(
    "SOCIAL_PUBLICATION_CLAIM_BATCH_SIZE",
    50,
    minimum=1,
)
SOURCE_VIDEO_LOCK_WAIT_SECONDS = _env_int(
    "SOURCE_VIDEO_LOCK_WAIT_SECONDS",
    180,
    minimum=1,
)
SOURCE_VIDEO_LOCK_TTL_SECONDS = _env_int(
    "SOURCE_VIDEO_LOCK_TTL_SECONDS",
    600,
    minimum=5,
)

# ---------------------------------------------------------------------------
# Credits - Billing pricing rules
# ---------------------------------------------------------------------------


def calculate_video_analysis_cost(duration_seconds: int) -> int:
    """
    Calculate analysis credits based on video duration.

    Rule: 1 credit per started minute (rounded up).
    """
    seconds = max(int(duration_seconds), 0)
    if seconds <= 0:
        return 0
    return int(ceil(seconds / 60))


CREDIT_COST_CLIP_GENERATION = 3
CREDIT_COST_CLIP_SMART_CLEANUP_SURCHARGE = 1
_SMART_CLEANUP_SURCHARGE_WAIVED_TIERS = {"basic", "pro", "enterprise"}


def smart_cleanup_surcharge_for_tier(tier: str | None) -> int:
    normalized_tier = str(tier or "").strip().lower()
    if normalized_tier in _SMART_CLEANUP_SURCHARGE_WAIVED_TIERS:
        return 0
    return int(CREDIT_COST_CLIP_SMART_CLEANUP_SURCHARGE)


def calculate_clip_generation_cost(
    smart_cleanup_enabled: bool,
    tier: str | None = None,
) -> int:
    """Return clip generation credit cost with optional tier-aware Smart Cleanup surcharge."""
    surcharge = smart_cleanup_surcharge_for_tier(tier) if smart_cleanup_enabled else 0
    return int(CREDIT_COST_CLIP_GENERATION) + surcharge


def normalize_clip_generation_credits(
    raw_credits: object,
    *,
    minimum_credits: int | None = None,
) -> int:
    """
    Normalize queued generation credits and enforce a minimum billable amount.

    This protects generation tasks from stale/tampered queued payloads that
    might otherwise resolve to zero/negative values.
    """
    try:
        parsed = int(raw_credits)
    except (TypeError, ValueError):
        parsed = 0

    minimum = int(
        CREDIT_COST_CLIP_GENERATION if minimum_credits is None else minimum_credits
    )
    return max(minimum, parsed)


def normalize_custom_clip_generation_credits(raw_credits: object) -> int:
    """
    Enforce that custom clips always consume credits.

    Custom clip generation should never resolve to zero (or negative) credits,
    even if a queued payload is stale or tampered.
    """
    return normalize_clip_generation_credits(raw_credits)


def calculate_custom_clip_generation_cost(
    smart_cleanup_enabled: bool,
    tier: str | None = None,
) -> int:
    """Return a custom-clip cost that is guaranteed to stay credit-consuming."""
    calculated = calculate_clip_generation_cost(
        smart_cleanup_enabled=smart_cleanup_enabled,
        tier=tier,
    )
    return normalize_custom_clip_generation_credits(calculated)


# ---------------------------------------------------------------------------
# Polar usage events (optional)
# ---------------------------------------------------------------------------
POLAR_ENV = os.getenv("POLAR_ENV", "sandbox")
POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_ORGANIZATION_ID = os.getenv("POLAR_ORGANIZATION_ID")
POLAR_USAGE_EVENTS_ENABLED = _env_bool("POLAR_USAGE_EVENTS_ENABLED", True)
POLAR_USAGE_EVENT_TIMEOUT_SECONDS = float(
    os.getenv("POLAR_USAGE_EVENT_TIMEOUT_SECONDS", "5")
)
POLAR_USAGE_EVENT_ANALYSIS_NAME = os.getenv(
    "POLAR_USAGE_EVENT_ANALYSIS_NAME",
    "clipscut.analysis.usage",
)
POLAR_USAGE_EVENT_GENERATION_NAME = os.getenv(
    "POLAR_USAGE_EVENT_GENERATION_NAME",
    "clipscut.generation.usage",
)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Clip-window retranscription can use a smaller model for better latency.
# Set to WHISPER_MODEL to keep a single-model behavior.
WHISPER_CLIP_MODEL = os.getenv("WHISPER_CLIP_MODEL", "tiny")
MAX_RETRIES = 3
YTDLP_SOCKET_TIMEOUT_SECONDS = _env_int(
    "YTDLP_SOCKET_TIMEOUT_SECONDS",
    20,
    minimum=1,
)
YTDLP_DOWNLOAD_RETRIES = _env_int("YTDLP_DOWNLOAD_RETRIES", 3, minimum=1)
YTDLP_FRAGMENT_RETRIES = _env_int("YTDLP_FRAGMENT_RETRIES", 3, minimum=1)
YTDLP_EXTRACTOR_RETRIES = _env_int("YTDLP_EXTRACTOR_RETRIES", 3, minimum=1)
YTDLP_COOKIES_FILE: str | None = (
    os.getenv("YTDLP_COOKIES_FILE", "").strip() or None
)
YTDLP_COOKIES_SOURCE_FILE: str | None = (
    os.getenv("YTDLP_COOKIES_SOURCE_FILE", "").strip() or None
)
YTDLP_PROXY: str | None = os.getenv("YTDLP_PROXY", "").strip() or None
YTDLP_POT_PROVIDER_URL: str | None = (
    os.getenv("YTDLP_POT_PROVIDER_URL", "").strip() or None
)
FFMPEG_THREADS = _env_int("FFMPEG_THREADS", 2, minimum=1)
RAW_VIDEO_STORAGE_BUCKET = (
    os.getenv("RAW_VIDEO_STORAGE_BUCKET", "raw-videos").strip() or "raw-videos"
)
YOUTUBE_SHORTS_MAX_DURATION_SECONDS = _env_int(
    "YOUTUBE_SHORTS_MAX_DURATION_SECONDS",
    180,
    minimum=1,
)
SOCIAL_ACCOUNT_ENCRYPTION_KEY = os.getenv("SOCIAL_ACCOUNT_ENCRYPTION_KEY")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
_ANALYZER_API_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def validate_env(extra: list[str] | None = None):
    """Check that critical env vars are set.  Call at process startup."""
    required = list(_REQUIRED_ENV) + (extra or [])
    missing = [v for v in required if not os.getenv(v)]
    if not any(os.getenv(name) for name in _ANALYZER_API_KEYS):
        missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)

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

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/video_clipper")
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 500))

# ---------------------------------------------------------------------------
# Workers - concurrency & timeouts
# ---------------------------------------------------------------------------
NUM_VIDEO_WORKERS = int(os.getenv("NUM_VIDEO_WORKERS", 2))
NUM_CLIP_WORKERS = int(os.getenv("NUM_CLIP_WORKERS", 2))
VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", 1800))  # 30 min
CLIP_JOB_TIMEOUT = int(os.getenv("CLIP_JOB_TIMEOUT", 1800))  # 30 min
RAW_VIDEO_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("RAW_VIDEO_CLEANUP_INTERVAL_SECONDS", 300)
)
CLIP_ASSET_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("CLIP_ASSET_CLEANUP_INTERVAL_SECONDS", 300)
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
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "ANTHROPIC_API_KEY"]


def validate_env(extra: list[str] | None = None):
    """Check that critical env vars are set.  Call at process startup."""
    required = list(_REQUIRED_ENV) + (extra or [])
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)

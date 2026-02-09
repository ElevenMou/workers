import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
# Workers – concurrency & timeouts
# ---------------------------------------------------------------------------
NUM_VIDEO_WORKERS = int(os.getenv("NUM_VIDEO_WORKERS", 2))
NUM_CLIP_WORKERS = int(os.getenv("NUM_CLIP_WORKERS", 2))
VIDEO_JOB_TIMEOUT = int(os.getenv("VIDEO_JOB_TIMEOUT", 1800))  # 30 min
CLIP_JOB_TIMEOUT = int(os.getenv("CLIP_JOB_TIMEOUT", 1800))  # 30 min

# ---------------------------------------------------------------------------
# Credits – Dynamic pricing based on duration
# ---------------------------------------------------------------------------


def calculate_video_analysis_cost(duration_seconds: int) -> int:
    """
    Calculate credits based on video duration.

    - 0-5 min:   3 credits
    - 5-15 min:  5 credits
    - 15-30 min: 8 credits
    - 30-60 min: 12 credits
    - 60+ min:   15 credits
    """
    minutes = duration_seconds / 60

    if minutes <= 5:
        return 3
    elif minutes <= 15:
        return 5
    elif minutes <= 30:
        return 8
    elif minutes <= 60:
        return 12
    else:
        return 15


CREDIT_COST_CLIP_GENERATION = 2

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]


def validate_env(extra: list[str] | None = None):
    """Check that critical env vars are set.  Call at process startup."""
    required = list(_REQUIRED_ENV) + (extra or [])
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        raise SystemExit(1)

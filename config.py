import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Storage
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/video_clipper")
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 500))


# Credits - Dynamic pricing based on duration
def calculate_video_analysis_cost(duration_seconds: int) -> int:
    """
    Calculate credits based on video duration
    - 0-5 min: 3 credits
    - 5-15 min: 5 credits
    - 15-30 min: 8 credits
    - 30-60 min: 12 credits
    - 60+ min: 15 credits
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

# Processing
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MAX_RETRIES = 3

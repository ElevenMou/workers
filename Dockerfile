# ---------------------------------------------------------------------------
# Clipry Workers - API + RQ worker pool
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies:
#   ffmpeg  - video processing (ffmpeg-python, yt-dlp)
#   git     - pip install openai-whisper from GitHub
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Install Python deps in two stages for smaller image -------------------
# 1) CPU-only PyTorch first (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- Copy application code -------------------------------------------------
COPY . .

# Create a non-root user for security
RUN useradd -m -r appuser

# Ensure the default temp directory exists and is writable by appuser
RUN mkdir -p /tmp/video_clipper && chown appuser:appuser /tmp/video_clipper

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8001

# Default: run the RQ worker pool
# (overridden to uvicorn for the api service in docker-compose)
CMD ["python", "main.py"]

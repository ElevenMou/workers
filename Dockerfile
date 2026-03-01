# ---------------------------------------------------------------------------
# Clipry Workers - API + RQ worker pool (multi-stage build)
# ---------------------------------------------------------------------------

# === Stage 1: Build dependencies ===
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only PyTorch first (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2: Runtime image ===
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Runtime-only system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m -r appuser

# Ensure the temp directory and yt-dlp runtime cookie path are writable by appuser
RUN mkdir -p /tmp/video_clipper/yt-dlp && chown -R appuser:appuser /tmp/video_clipper

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8001

# Ensure Docker sends SIGTERM for graceful shutdown
STOPSIGNAL SIGTERM

# Default: run the RQ worker pool
# (overridden to gunicorn for the api service in docker-compose)
CMD ["python", "main.py"]

## Workers Quickstart

Follow these steps to run the worker stack locally with MinIO-only object storage.

### 1) Create and activate a virtual environment

Python `3.11.x` is required.

```bash
py -3.11 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install ffmpeg

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: download from [https://ffmpeg.org](https://ffmpeg.org) and add it to `PATH`

### 4) Set up Supabase

1. Create a Supabase project.
2. Apply the SQL schema and migrations from the repo.
3. Copy the project URL and service-role key from `Settings -> API`.

### 5) Set up MinIO

Create these buckets:
- `generated-clips`
- `raw-videos`
- `avatars`
- `layouts`

Recommended policies:
- `avatars`: public read
- `generated-clips`: private
- `raw-videos`: private
- `layouts`: private

### 6) Set up Redis

```bash
redis-server
redis-cli ping
```

### 7) Configure `workers/.env`

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=replace-with-supabase-service-role-key
SUPABASE_JWT_SECRET=replace-with-supabase-jwt-secret
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=changeme
TEMP_DIR=/tmp/video_clipper
OPENAI_API_KEY=sk-...
WHISPER_MODEL=base
MEDIA_STORAGE_PROVIDER=minio
WORKER_PUBLIC_BASE_URL=http://localhost:8001
WORKER_MEDIA_SIGNING_SECRET=replace-with-worker-media-signing-secret
WORKER_INTERNAL_API_TOKEN=replace-with-worker-internal-token
MINIO_ENDPOINT=localhost:9000
MINIO_PUBLIC_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_REGION=us-east-1
MINIO_CLIPS_BUCKET=generated-clips
MINIO_RAW_VIDEOS_BUCKET=raw-videos
MINIO_AVATARS_BUCKET=avatars
MINIO_LAYOUTS_BUCKET=layouts
NUM_VIDEO_WORKERS=2
NUM_CLIP_WORKERS=2
NUM_SOCIAL_WORKERS=1
FFMPEG_THREADS=2
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

### 8) Test the database connection

```python
from utils.supabase_client import supabase

result = supabase.table("videos").select("*").limit(1).execute()
print("Database connected")
print(result)
```

### 9) Test Redis

```python
from utils.redis_client import redis_conn

redis_conn.set("test", "hello")
print(redis_conn.get("test"))
```

### 10) Start the stack

```bash
python main.py
```

### 11) Smoke-test media storage

Validate this end-to-end flow:
1. Upload an avatar.
2. Upload layout assets.
3. Analyze a video and generate a clip.
4. Preview and download the clip.
5. Delete the clip and the source video.

Only MinIO buckets should change during media operations. Supabase remains the source of truth for auth, tables, and RPCs only.

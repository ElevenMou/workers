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
WORKER_PUBLIC_BASE_URL=http://localhost:7050
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

## Production Behind Cloudflare Tunnel

Use the worker stack behind `cloudflared` when serving `api.clipscut.pro`.

1. Copy `workers/.env.production.example` to `workers/.env.production`.
2. Keep `CADDY_DOMAIN=:7050` and `WORKER_PUBLIC_BASE_URL=https://api.clipscut.pro`.
3. Create or inspect the Cloudflare tunnel and note its UUID with `cloudflared tunnel list`.
4. Copy `workers/cloudflared/api-tunnel.yml.example` to `/etc/cloudflared/api-clipscut.yml` and replace `<TUNNEL_UUID>` with the actual tunnel UUID.
5. Validate the ingress rules with `cloudflared tunnel ingress validate --config /etc/cloudflared/api-clipscut.yml`.
6. Copy `workers/cloudflared/cloudflared-api.service.example` to `/etc/systemd/system/cloudflared-api.service`.
7. Enable the tunnel service with `sudo systemctl enable --now cloudflared-api`.
8. If the service does not come up, run `sudo bash scripts/diagnose_cloudflared_api.sh`.
9. Start or update the stack with `cd workers && sudo bash run_server.sh`.
10. Verify the local origin with `curl http://127.0.0.1:7050/ready`.
11. Verify the public origin with `curl https://api.clipscut.pro/ready`.
12. Run `LOCAL_API_ORIGIN=http://127.0.0.1:7050 bash ../scripts/smoke-test.sh --api-only`.

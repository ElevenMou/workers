## Workers Quickstart

Follow these steps to bring the worker stack up locally.

### 1) Create and activate a virtual environment

**important: python version 3.11.x**

```bash
py -3.11 -m venv .venv
# macOS/Linux
source venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install ffmpeg

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: download from [https://ffmpeg.org](https://ffmpeg.org) and add to PATH

### 4) Set up Supabase

1. Create a project at [https://supabase.com](https://supabase.com).
2. Run the SQL schema from the main repo to create tables/functions.
3. In `Settings → API`, copy your `Project URL` and `service_role` key.
4. Create storage buckets:
  - `generated-clips` (public)
  - `thumbnails` (public)
  - `raw-videos` (private)

### 5) Set up Redis

- macOS: `brew install redis`
- Ubuntu: `sudo apt-get install redis`
- Windows: use Docker or WSL

Start Redis and verify:

```bash
redis-server
redis-cli ping  # should return PONG
```

### 6) Configure environment

Create `workers/.env` with your values:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
REDIS_HOST=localhost
REDIS_PORT=6379
TEMP_DIR=/tmp/video_clipper
WHISPER_MODEL=base
OPENAI_API_KEY=sk-openai-your-key
OPENAI_ANALYZER_MODEL=gpt-4.1-mini
OPENAI_ANALYZER_FALLBACK_MODELS=gpt-4.1-mini,gpt-4.1
WORKER_SCALE_ADMIN_USER_IDS=uuid-admin-1,uuid-admin-2
PURGE_QUEUED_JOBS_ON_START=false
FAIL_STARTED_JOBS_ON_START=true
FAIL_PROCESSING_ROWS_ON_START=true
RAW_VIDEO_CLEANUP_INTERVAL_SECONDS=300
RAW_VIDEO_STORAGE_BUCKET=raw-videos
SUPERVISOR_ROLE=worker
WORKER_INSTANCE_ID=node-a
MAINTENANCE_LEADER_LOCK_TTL_SECONDS=15
MAINTENANCE_LEADER_RENEW_SECONDS=5
FFMPEG_THREADS=2
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
POLAR_ACCESS_TOKEN=polar_org_token_with_events_write_scope
POLAR_ENV=sandbox
POLAR_USAGE_EVENTS_ENABLED=true
POLAR_USAGE_EVENT_ANALYSIS_NAME=clipscut.analysis.usage
POLAR_USAGE_EVENT_GENERATION_NAME=clipscut.generation.usage
POLAR_USAGE_EVENT_TIMEOUT_SECONDS=5
# Optional when not using an organization token:
# POLAR_ORGANIZATION_ID=org_xxx
```

Notes:
- `PURGE_QUEUED_JOBS_ON_START` now defaults to `false` and should stay `false` in production.
- Legacy `ANTHROPIC_API_KEY` and `CLAUDE_ANALYZER_*` env vars are still accepted during the migration window, but new deployments should use the `OPENAI_*` names above.
- Run a dedicated `maintenance-supervisor` instance with `SUPERVISOR_ROLE=maintenance`.
- Use `WORKER_SCALE_ADMIN_USER_IDS` (comma-separated) to allow `/workers/scale` admin access.
- Mutating API routes (`/videos/analyze`, `/clips/generate`, `/clips/custom`, `/workers/scale`) now require a valid `Authorization: Bearer <supabase-jwt>` header.
- Polar usage event ingestion is non-blocking. If Polar is unavailable, jobs still complete and credits are still charged.

### 7) Test database connection

Create `workers/test_db.py`:

```python
from utils.supabase_client import supabase

result = supabase.table("videos").select("*").limit(1).execute()
print("✓ Database connected")
print(result)
```

Run: `python test_db.py`

### 8) Test Redis connection

Create `workers/test_redis.py`:

```python
from utils.redis_client import redis_conn

redis_conn.set("test", "hello")
value = redis_conn.get("test")
print(f"✓ Redis connected: {value}")
```

Run: `python test_redis.py`

### 9) Create a test user

In Supabase SQL Editor:

```sql
-- Create a test user (or sign up via Supabase Auth)
-- Get the user_id from auth.users
SELECT id, email FROM auth.users;
```

### 10) Manual video analysis smoke test

Create `workers/test_analyze.py`:

```python
import uuid
from tasks.analyze_video import analyze_video_task
from utils.supabase_client import supabase

job_data = {
    "jobId": str(uuid.uuid4()),
    "videoId": str(uuid.uuid4()),
    "userId": "your-user-id-from-step-9",  # replace
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "numClips": 3,
}

# Seed Supabase
supabase.table("videos").insert({
    "id": job_data["videoId"],
    "user_id": job_data["userId"],
    "url": job_data["url"],
    "status": "pending",
}).execute()

supabase.table("jobs").insert({
    "id": job_data["jobId"],
    "user_id": job_data["userId"],
    "video_id": job_data["videoId"],
    "type": "analyze_video",
    "status": "queued",
}).execute()

# Grant test credits
supabase.rpc("grant_credits", {
    "p_user_id": job_data["userId"],
    "p_amount": 20,
    "p_type": "bonus",
    "p_description": "Test credits",
}).execute()

print("Starting video analysis...")
analyze_video_task(job_data)
print("✓ Analysis complete!")
```

Run: `python test_analyze.py`

### 11) Start the worker (production mode)

```bash
python main.py
```

Expected logs:

```
Starting worker...
Queues: ['video-processing', 'clip-generation']
Worker started...
```

### 12) Queue a job via Redis

Create `workers/queue_job.py`:

```python
import uuid
from utils.redis_client import video_queue
from utils.supabase_client import supabase

user_id = "your-user-id"
video_id = str(uuid.uuid4())
job_id = str(uuid.uuid4())

# Insert records
supabase.table("videos").insert({
    "id": video_id,
    "user_id": user_id,
    "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo"
    "status": "pending",
}).execute()

supabase.table("jobs").insert({
    "id": job_id,
    "user_id": user_id,
    "video_id": video_id,
    "type": "analyze_video",
    "status": "queued",
}).execute()

# Queue the job
video_queue.enqueue(
    "tasks.analyze_video.analyze_video_task",
    {
        "jobId": job_id,
        "videoId": video_id,
        "userId": user_id,
        "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "numClips": 3,
    },
)

print(f"✓ Job queued: {job_id}")
```

Run: `python queue_job.py` and watch the worker logs.

### 13) Monitor progress in Supabase

```sql
-- Job status
SELECT * FROM jobs WHERE id = 'your-job-id';

-- Video status
SELECT * FROM videos WHERE id = 'your-video-id';

-- Generated clips
SELECT * FROM clips WHERE video_id = 'your-video-id';

-- Credit balance
SELECT * FROM credit_balances WHERE user_id = 'your-user-id';
```

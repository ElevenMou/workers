import uuid

from config import CREDIT_COST_CLIP_GENERATION
from services.video_downloader import VideoDownloader
from tasks.generate_clip import generate_clip_task
from utils.supabase_client import supabase

# TODO: replace with a real user id from your Supabase auth.users table
USER_ID = "994ad2b6-d875-4d14-a493-28697a578862"

# A short, publicly accessible video keeps tests fast
VIDEO_URL = "https://www.youtube.com/watch?v=FWkVBjcVw18"  # Me at the zoo


def seed_records():
    """Download a video and seed the minimal Supabase rows needed for clip generation."""
    if USER_ID.startswith("replace"):
        raise ValueError(
            "Set USER_ID to a valid Supabase user id before running the test."
        )

    downloader = VideoDownloader()

    job_id = str(uuid.uuid4())
    video_id = str(uuid.uuid4())
    clip_id = str(uuid.uuid4())

    video_data = downloader.download(VIDEO_URL, video_id)
    raw_video_path = video_data["path"]
    duration = int(video_data.get("duration") or 120)

    # Use a ~65s segment to satisfy duration checks enforced in DB
    clip_start = 5.0
    clip_duration = 65.0
    clip_end = min(duration - 1, clip_start + clip_duration)

    if clip_end - clip_start < 60:
        raise ValueError(
            f"Video too short for test clip: duration={duration}s. "
            "Use a longer URL or reduce the minimum clip length."
        )

    # Seed video row
    supabase.table("videos").insert(
        {
            "id": video_id,
            "user_id": USER_ID,
            "url": VIDEO_URL,
            "title": video_data.get("title"),
            "duration_seconds": duration,
            "status": "completed",
            "raw_video_path": raw_video_path,
        }
    ).execute()

    # Seed clip row
    supabase.table("clips").insert(
        {
            "id": clip_id,
            "video_id": video_id,
            "user_id": USER_ID,
            "start_time": clip_start,
            "end_time": clip_end,
            "title": "Test clip",
            "background_style": "blur",
            "status": "pending",
        }
    ).execute()

    # Seed job row
    supabase.table("jobs").insert(
        {
            "id": job_id,
            "user_id": USER_ID,
            "video_id": video_id,
            "type": "generate_clip",
            "status": "queued",
        }
    ).execute()

    # Ensure enough credits for clip generation
    supabase.rpc(
        "grant_credits",
        {
            "p_user_id": USER_ID,
            "p_amount": CREDIT_COST_CLIP_GENERATION,
            "p_type": "bonus",
            "p_description": "Test clip generation credits",
        },
    ).execute()

    return {"jobId": job_id, "clipId": clip_id, "userId": USER_ID}


if __name__ == "__main__":
    print("Seeding Supabase and downloading test video...")
    job_data = seed_records()
    print("Starting clip generation...")
    generate_clip_task(job_data)
    print("✓ Clip generation complete!")

import uuid
from tasks.analyze_video import analyze_video_task
from utils.supabase_client import supabase

job_data = {
    "jobId": str(uuid.uuid4()),
    "videoId": str(uuid.uuid4()),
    "userId": "994ad2b6-d875-4d14-a493-28697a578862",
    "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
    "numClips": 3,
}

# Seed Supabase
supabase.table("videos").insert(
    {
        "id": job_data["videoId"],
        "user_id": job_data["userId"],
        "url": job_data["url"],
        "status": "pending",
    }
).execute()

supabase.table("jobs").insert(
    {
        "id": job_data["jobId"],
        "user_id": job_data["userId"],
        "video_id": job_data["videoId"],
        "type": "analyze_video",
        "status": "queued",
    }
).execute()

# Grant test credits
supabase.rpc(
    "grant_credits",
    {
        "p_user_id": job_data["userId"],
        "p_amount": 20,
        "p_type": "bonus",
        "p_description": "Test credits",
    },
).execute()

print("Starting video analysis...")
analyze_video_task(job_data)
print("✓ Analysis complete!")

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def update_job_status(job_id: str, status: str, progress: int, error: str = None):
    """Update job status and progress"""
    data = {"status": status, "progress": progress}

    if status == "processing" and progress == 0:
        data["started_at"] = "now()"
    elif status in ["completed", "failed"]:
        data["completed_at"] = "now()"

    if error:
        data["error_message"] = error

    supabase.table("jobs").update(data).eq("id", job_id).execute()


def update_video_status(video_id: str, status: str, **kwargs):
    """Update video record"""
    data = {"status": status}
    data.update(kwargs)
    supabase.table("videos").update(data).eq("id", video_id).execute()

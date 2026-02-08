import os
import traceback
from services.clip_generator import ClipGenerator
from utils.supabase_client import supabase, update_job_status
from config import CREDIT_COST_CLIP_GENERATION, TEMP_DIR


def generate_clip_task(job_data: dict):
    """Generate final clip with overlay"""
    job_id = job_data["jobId"]
    clip_id = job_data["clipId"]
    user_id = job_data["userId"]

    generator = ClipGenerator()

    try:
        # Get clip and video details
        clip = (
            supabase.table("clips")
            .select("*, videos(*)")
            .eq("id", clip_id)
            .single()
            .execute()
            .data
        )

        # Check credits
        result = supabase.rpc(
            "has_credits",
            {"p_user_id": user_id, "p_amount": CREDIT_COST_CLIP_GENERATION},
        ).execute()

        if not result.data:
            raise Exception("Insufficient credits")

        update_job_status(job_id, "processing", 0)

        # Update clip status
        supabase.table("clips").update({"status": "generating"}).eq(
            "id", clip_id
        ).execute()

        # Generate clip
        result = generator.generate(
            video_path=clip["videos"]["raw_video_path"],
            clip_id=clip_id,
            start_time=float(clip["start_time"]),
            end_time=float(clip["end_time"]),
            title=clip["title"],
            background_style=clip["background_style"],
        )

        update_job_status(job_id, "processing", 90)

        # Upload to Supabase Storage
        storage_path = f"clips/{clip_id}.mp4"
        thumbnail_path = f"thumbnails/{clip_id}.jpg"

        with open(result["clip_path"], "rb") as f:
            supabase.storage.from_("generated-clips").upload(storage_path, f)

        with open(result["thumbnail_path"], "rb") as f:
            supabase.storage.from_("thumbnails").upload(thumbnail_path, f)

        # Update clip record
        supabase.table("clips").update(
            {
                "status": "completed",
                "storage_path": storage_path,
                "thumbnail_path": thumbnail_path,
                "file_size_bytes": result["file_size"],
            }
        ).eq("id", clip_id).execute()

        # Charge credits
        supabase.rpc(
            "charge_credits",
            {
                "p_user_id": user_id,
                "p_amount": CREDIT_COST_CLIP_GENERATION,
                "p_type": "clip_generation",
                "p_description": f'Clip generation: {clip["title"][:50]}',
                "p_clip_id": clip_id,
            },
        ).execute()

        update_job_status(job_id, "completed", 100)

        # Cleanup temp files
        os.remove(result["clip_path"])
        os.remove(result["thumbnail_path"])

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating clip: {error_msg}")
        print(traceback.format_exc())

        update_job_status(job_id, "failed", 0, error_msg)
        supabase.table("clips").update(
            {"status": "failed", "error_message": error_msg}
        ).eq("id", clip_id).execute()

        raise

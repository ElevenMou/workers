import os
import traceback
from datetime import datetime, timedelta
from services.video_downloader import VideoDownloader
from services.transcriber import Transcriber
from services.ai_analyzer import AIAnalyzer
from utils.supabase_client import supabase, update_job_status, update_video_status
from config import calculate_video_analysis_cost


def analyze_video_task(job_data: dict):
    """Main task for video analysis"""
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    user_id = job_data["userId"]
    url = job_data["url"]
    num_clips = job_data.get("numClips", 5)

    downloader = VideoDownloader()
    transcriber = Transcriber()
    analyzer = AIAnalyzer()

    video_path = None
    audio_path = None

    try:
        update_job_status(job_id, "processing", 0)
        update_video_status(video_id, "downloading")

        # 1. Download video
        print(f"Downloading video: {url}")
        video_data = downloader.download(url, video_id)
        video_path = video_data["path"]
        update_job_status(job_id, "processing", 20)

        # Calculate credits based on duration
        duration_seconds = int(video_data["duration"])
        credits_required = calculate_video_analysis_cost(duration_seconds)

        print(f"Video duration: {duration_seconds}s (~{duration_seconds/60:.1f}min)")
        print(f"Credits required: {credits_required}")

        # Check credits
        result = supabase.rpc(
            "has_credits", {"p_user_id": user_id, "p_amount": credits_required}
        ).execute()

        if not result.data:
            raise Exception(f"Insufficient credits. Required: {credits_required}")

        # Update video metadata
        update_video_status(
            video_id,
            "analyzing",
            title=video_data["title"],
            duration_seconds=duration_seconds,
            thumbnail_url=video_data["thumbnail"],
            platform=video_data["platform"],
            external_id=video_data.get("external_id"),
            raw_video_path=video_path,
            raw_video_expires_at=(datetime.now() + timedelta(hours=24)).isoformat(),
        )

        # 2. Get transcript
        transcript = None
        if video_data["platform"] == "youtube" and video_data.get("external_id"):
            print("Attempting to get YouTube transcript...")
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                print("✓ Got transcript from YouTube")
                update_job_status(job_id, "processing", 60)

        if not transcript:
            print("Extracting audio for Whisper transcription...")
            audio_path = downloader.extract_audio(video_path)
            update_job_status(job_id, "processing", 40)

            print("Transcribing with Whisper...")
            transcript = transcriber.transcribe(audio_path)
            transcript["source"] = "whisper"
            update_job_status(job_id, "processing", 60)

        # 3. Analyze with Claude Opus 4
        print(f"Analyzing for {num_clips} clips with Claude Opus 4...")
        clips = analyzer.find_best_clips(transcript, num_clips=num_clips)
        update_job_status(job_id, "processing", 90)

        # 4. Save results
        print("Saving results...")
        supabase.table("videos").update(
            {"status": "completed", "transcript": transcript}
        ).eq("id", video_id).execute()

        # Insert clip suggestions
        for clip in clips:
            supabase.table("clips").insert(
                {
                    "video_id": video_id,
                    "user_id": user_id,
                    "start_time": clip["start"],
                    "end_time": clip["end"],
                    "title": clip["title"],
                    "ai_score": clip["score"],
                    "transcript_excerpt": clip["text"][:500],
                    "status": "pending",
                }
            ).execute()

        # Charge credits (dynamic amount)
        supabase.rpc(
            "charge_credits",
            {
                "p_user_id": user_id,
                "p_amount": credits_required,
                "p_type": "video_analysis",
                "p_description": f'Video analysis ({duration_seconds/60:.1f}min): {video_data["title"][:50]}',
                "p_video_id": video_id,
            },
        ).execute()

        supabase.table("videos").update(
            {"credits_charged": credits_required, "clip_count": len(clips)}
        ).eq("id", video_id).execute()

        update_job_status(job_id, "completed", 100)
        print(f"Video analysis completed: {video_id}")

    except Exception as e:
        error_msg = str(e)
        print(f"Error analyzing video: {error_msg}")
        print(traceback.format_exc())

        update_job_status(job_id, "failed", 0, error_msg)
        update_video_status(video_id, "failed", error_message=error_msg)

        raise

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

from __future__ import annotations

import api_app.helpers as api_helpers


def test_enqueue_or_fail_passes_job_timeout_seconds_to_enqueue_job(monkeypatch):
    enqueue_calls: list[dict] = []

    monkeypatch.setattr(
        api_helpers,
        "enqueue_job",
        lambda queue_name, task_path, job_data, *, job_id, job_timeout_seconds=None: enqueue_calls.append(
            {
                "queue_name": queue_name,
                "task_path": task_path,
                "job_data": dict(job_data),
                "job_id": job_id,
                "job_timeout_seconds": job_timeout_seconds,
            }
        ),
    )

    api_helpers.enqueue_or_fail(
        queue_name="video-processing-priority",
        task_path="tasks.videos.analyze.analyze_video_task",
        job_data={"jobId": "job-1"},
        job_timeout_seconds=3210,
        job_id="job-1",
        user_id="user-1",
        job_type="analyze_video",
        video_id="video-1",
    )

    assert enqueue_calls == [
        {
            "queue_name": "video-processing-priority",
            "task_path": "tasks.videos.analyze.analyze_video_task",
            "job_data": {"jobId": "job-1"},
            "job_id": "job-1",
            "job_timeout_seconds": 3210,
        }
    ]

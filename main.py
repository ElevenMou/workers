from rq import Worker
from utils.redis_client import redis_conn, video_queue, clip_queue
from tasks.analyze_video import analyze_video_task
from tasks.generate_clip import generate_clip_task
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    print("Starting worker...")
    print(f"Queues: {[q.name for q in [video_queue, clip_queue]]}")

    worker = Worker([video_queue, clip_queue], connection=redis_conn)

    worker.work(with_scheduler=True)

import redis
from rq import Queue
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

video_queue = Queue("video-processing", connection=redis_conn)
clip_queue = Queue("clip-generation", connection=redis_conn)

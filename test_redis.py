from utils.redis_client import redis_conn

redis_conn.set("test", "hello")
value = redis_conn.get("test")
print(f"Redis connected: {value}")

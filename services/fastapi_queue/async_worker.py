"""Async Redis worker — replaces rq_worker.py.

Uses redis.asyncio (built into redis-py 4+) with BLPOP so the process never
spins: it blocks in the kernel until a job arrives, processes it in a thread
pool (so the event loop stays free), then immediately dequeues the next job.

A single worker process can saturate all CPU cores through the ThreadPoolExecutor
while handling many concurrent jobs. This removes all RQ bookkeeping overhead
(~15 Redis ops per job → 3 ops: BLPOP + LPUSH result + EXPIRE).

Usage:
    python services/fastapi_queue/async_worker.py
    python services/fastapi_queue/async_worker.py --redis-url redis://localhost:6379 --threads 4
"""

import argparse
import asyncio
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from redis.asyncio import Redis

from services.fastapi_queue.inference_task import _get_model, predict_batch

JOBS_KEY = "inference:jobs"
RESULT_TTL = 60  # seconds

parser = argparse.ArgumentParser(description="Async Redis inference worker")
parser.add_argument("--redis-url", default="redis://localhost:6379")
parser.add_argument("--threads", type=int, default=4,
                    help="ThreadPoolExecutor size for CPU-bound inference")
args = parser.parse_args()


async def handle_job(redis: Redis, executor: ThreadPoolExecutor, raw: bytes) -> None:
    job = pickle.loads(raw)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, predict_batch, job["vessels"])
    result_key = f"inference:result:{job['id']}"
    await redis.lpush(result_key, pickle.dumps(result))
    await redis.expire(result_key, RESULT_TTL)


async def main() -> None:
    # Warm up model in the main process before forking threads
    _get_model()

    redis = Redis.from_url(args.redis_url)
    executor = ThreadPoolExecutor(max_workers=args.threads)
    print(f"Async worker ready — {args.threads} inference threads, listening on {args.redis_url}")

    while True:
        item = await redis.blpop(JOBS_KEY, timeout=0)  # blocks until a job arrives
        # Fire-and-forget: immediately dequeue the next job without waiting for this one
        asyncio.create_task(handle_job(redis, executor, item[1]))


if __name__ == "__main__":
    asyncio.run(main())

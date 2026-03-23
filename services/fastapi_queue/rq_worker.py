"""RQ worker process — run this alongside the FastAPI queue server.

Usage:
    python services/fastapi_queue/rq_worker.py
    python services/fastapi_queue/rq_worker.py --redis-url redis://localhost:6379
"""

import argparse
import os
import sys

# project root on sys.path so services.fastapi_queue.inference_task is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

parser = argparse.ArgumentParser(description="RQ inference worker")
parser.add_argument("--redis-url", default="redis://localhost:6379")
args = parser.parse_args()

if __name__ == "__main__":
    conn = Redis.from_url(args.redis_url)
    # SimpleWorker runs jobs in-process (no fork per job).
    # This means the model loaded by _get_model() is cached for the lifetime
    # of the worker process rather than reloaded from disk on every job.
    worker = SimpleWorker([Queue("default", connection=conn)], connection=conn)
    print(f"RQ SimpleWorker started, listening on {args.redis_url}")
    worker.work()

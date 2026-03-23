"""FastAPI inference server backed by a Redis/RQ job queue.

Each request is enqueued as an RQ job; the server async-polls Redis for the
result and returns it synchronously to the caller. End-to-end latency captures
the full queue overhead, making it directly comparable to the direct server.

Requires a running Redis instance and at least one RQ worker:
    python services/fastapi_queue/rq_worker.py

Usage:
    python services/fastapi_queue/server.py
    python services/fastapi_queue/server.py --port 8002 --redis-url redis://localhost:6379
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException, Request, Response
from redis import Redis
from rq import Queue

try:
    import orjson

    class ORJSONResponse(Response):
        media_type = "application/json"

        def render(self, content: object) -> bytes:
            return orjson.dumps(content)

    DEFAULT_RESPONSE = ORJSONResponse
except ImportError:
    DEFAULT_RESPONSE = None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FastAPI + RQ inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--redis-url", default="redis://localhost:6379")
parser.add_argument("--job-timeout", type=int, default=30)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
_redis = Redis.from_url(args.redis_url)
_queue = Queue(connection=_redis)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app_kwargs = {}
if DEFAULT_RESPONSE is not None:
    app_kwargs["default_response_class"] = DEFAULT_RESPONSE

app = FastAPI(title="Vessel Track Inference (Queue)", version="1.0", **app_kwargs)


async def _wait_for_job(job, poll_interval: float = 0.002):
    """Async-poll Redis until the job result is available."""
    deadline = time.monotonic() + args.job_timeout
    while time.monotonic() < deadline:
        job.refresh()
        if job.result is not None:
            return job.result
        if job.is_failed:
            raise HTTPException(status_code=500, detail="Inference job failed")
        await asyncio.sleep(poll_interval)
    raise HTTPException(status_code=504, detail="Inference job timed out")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    job = _queue.enqueue(
        "services.fastapi_queue.inference_task.predict_batch",
        [{"history": body["history"]}],
        job_timeout=args.job_timeout,
    )
    predictions = await _wait_for_job(job)
    return {"prediction": predictions[0]}


@app.post("/predict_batch")
async def predict_batch(request: Request):
    body = await request.json()
    job = _queue.enqueue(
        "services.fastapi_queue.inference_task.predict_batch",
        body["vessels"],
        job_timeout=args.job_timeout,
    )
    predictions = await _wait_for_job(job)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

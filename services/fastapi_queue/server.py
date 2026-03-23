"""FastAPI inference server backed by a Redis job queue (custom async implementation).

Each request is enqueued with RPUSH; the server then does a single BLPOP on the
result key — no polling loop, no sleep overhead. The worker (async_worker.py)
does the symmetric BLPOP on the jobs queue and LPUSH on the result key.

Redis ops per request: 2 on the server (RPUSH + BLPOP), 3 on the worker
(BLPOP + LPUSH + EXPIRE) — 5 total vs RQ's ~15+ bookkeeping calls.

Requires a running Redis instance and the async worker:
    python services/fastapi_queue/async_worker.py

Usage:
    python services/fastapi_queue/server.py
    python services/fastapi_queue/server.py --port 8002 --redis-url redis://localhost:6379
"""

import argparse
import os
import pickle
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException, Request, Response
from redis.asyncio import Redis

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
parser = argparse.ArgumentParser(description="FastAPI + async Redis queue inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--redis-url", default="redis://localhost:6379")
parser.add_argument("--job-timeout", type=int, default=30)
args = parser.parse_args()

JOBS_KEY = "inference:jobs"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app_kwargs = {}
if DEFAULT_RESPONSE is not None:
    app_kwargs["default_response_class"] = DEFAULT_RESPONSE

app = FastAPI(title="Vessel Track Inference (Queue)", version="1.0", **app_kwargs)

_redis: Redis | None = None


@app.on_event("startup")
async def startup():
    global _redis
    _redis = Redis.from_url(args.redis_url)


@app.on_event("shutdown")
async def shutdown():
    if _redis:
        await _redis.aclose()


async def _enqueue_and_wait(vessels: list) -> list:
    job_id = uuid.uuid4().hex
    payload = pickle.dumps({"id": job_id, "vessels": vessels})
    result_key = f"inference:result:{job_id}"

    await _redis.rpush(JOBS_KEY, payload)

    # BLPOP blocks until the worker pushes the result — no polling, no sleep
    item = await _redis.blpop(result_key, timeout=args.job_timeout)
    if item is None:
        raise HTTPException(status_code=504, detail="Inference job timed out")

    return pickle.loads(item[1])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    predictions = await _enqueue_and_wait([{"history": body["history"]}])
    return {"prediction": predictions[0]}


@app.post("/predict_batch")
async def predict_batch(request: Request):
    body = await request.json()
    predictions = await _enqueue_and_wait(body["vessels"])
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

"""FastAPI inference server that offloads prediction to an RQ job queue.

Instead of running inference in the request handler, this server enqueues
a prediction job to Redis via RQ and polls for the result asynchronously.
The actual inference runs in a separate rq worker process.

Usage:
    python server_fastapi_rq.py
    python server_fastapi_rq.py --port 8001 --redis-url redis://localhost:6379

Requires:
    - Redis running (default: redis://localhost:6379)
    - At least one RQ worker: rq worker inference --url redis://localhost:6379
"""

import argparse
import asyncio

import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request, Response
from rq import Queue
from rq.job import Job, JobStatus

try:
    import orjson

    def _orjson_dumps(obj: object) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)

    class ORJSONResponse(Response):
        media_type = "application/json"

        def render(self, content: object) -> bytes:
            return _orjson_dumps(content)

    DEFAULT_RESPONSE = ORJSONResponse
except ImportError:
    DEFAULT_RESPONSE = None  # fall back to FastAPI default

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FastAPI + RQ inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--redis-url", default="redis://localhost:6379")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Redis / RQ
# ---------------------------------------------------------------------------
_redis_conn = redis_lib.from_url(args.redis_url)
_queue = Queue("inference", connection=_redis_conn)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app_kwargs = {}
if DEFAULT_RESPONSE is not None:
    app_kwargs["default_response_class"] = DEFAULT_RESPONSE

app = FastAPI(title="Vessel Track Inference (RQ)", version="1.0", **app_kwargs)

_POLL_INTERVAL = 0.005  # 5 ms — trade-off between latency and CPU spin


async def _wait_for_job(job_id: str) -> list:
    """Async-poll Redis until the job finishes; return the result."""
    loop = asyncio.get_running_loop()
    while True:
        job: Job = await loop.run_in_executor(
            None, lambda: Job.fetch(job_id, connection=_redis_conn)
        )
        status: JobStatus = job.get_status()
        if status == JobStatus.FINISHED:
            return job.result
        if status in (JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELED):
            raise HTTPException(status_code=500, detail=f"Job {status.value}")
        await asyncio.sleep(_POLL_INTERVAL)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict_batch")
async def predict_batch(request: Request):
    body = await request.json()
    vessels = body["vessels"]
    histories = [
        [
            [p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]]
            for p in v["history"]
        ]
        for v in vessels
    ]
    # Enqueue using dotted string path so the server process never loads PyTorch
    job = _queue.enqueue("worker.predict_batch", histories, job_timeout=30)
    result = await _wait_for_job(job.id)
    return {"predictions": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_fastapi_rq:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="warning",
    )

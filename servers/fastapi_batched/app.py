"""FastAPI inference server -- in-process dynamic batching (torch runtime).

POST /predict  -- run inference, return result inline (same API as `fastapi`)
GET  /health   -- readiness check

Request flow:
    async handler -> bounded asyncio.Queue -> batcher task -> 1 inference thread

The batcher takes whatever is waiting in the queue (up to MAX_BATCH samples),
concatenates it into one tensor and runs a single forward pass on a dedicated
thread, so the event loop never blocks. At low load batches are size 1 with no
added delay; under load they grow while the previous forward pass is running.

Scale out with processes, not threads: WORKERS uvicorn processes each run their
own batcher, so aim for WORKERS x TORCH_NUM_THREADS ~= cores. For this small
LSTM, extra intra-op threads barely help and oversubscription hurts.

Tunables (env vars):
    WORKERS             uvicorn worker processes                      (default: 1)
    TORCH_NUM_THREADS   intra-op threads per forward pass             (default: 1)
    MAX_BATCH           max samples per forward pass                  (default: 64)
    MAX_QUEUE           max queued requests before returning 503      (default: 1024)
    REQUEST_TIMEOUT_S   per-request deadline before returning 504     (default: 1.0)
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import HISTORY_FEATURES, HISTORY_STEPS, INPUT_SIZE, VesselTrackPredictor

_DOCKER_WEIGHTS = Path("/app/weights/model.pt")
_LOCAL_WEIGHTS = (
    Path(__file__).resolve().parent.parent.parent / "model" / "weights" / "model.pt"
)
WEIGHTS = _DOCKER_WEIGHTS if _DOCKER_WEIGHTS.exists() else _LOCAL_WEIGHTS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "1024"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "1.0"))


def load_model() -> VesselTrackPredictor:
    model = VesselTrackPredictor().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location=DEVICE))
    model.eval()
    return model


class Batcher:
    def __init__(self, model: VesselTrackPredictor, max_batch: int, max_queue: int):
        self.model = model
        self.max_batch = max_batch
        self.q: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue(
            maxsize=max_queue
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

    def _infer(self, x: np.ndarray) -> np.ndarray:
        """Runs on the inference thread. x: (N, 30, 5) -> (N, 15, 2)."""
        with torch.inference_mode():
            return self.model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()

    async def submit(self, x: np.ndarray) -> np.ndarray:
        """Raises asyncio.QueueFull when saturated (caller maps it to 503)."""
        fut = asyncio.get_running_loop().create_future()
        self.q.put_nowait((x, fut))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.q.get()]
            n = len(items[0][0])
            while n < self.max_batch and not self.q.empty():
                item = self.q.get_nowait()
                items.append(item)
                n += len(item[0])

            # Drop requests whose caller already timed out.
            items = [(x, f) for x, f in items if not f.done()]
            if not items:
                continue

            try:
                ys = await loop.run_in_executor(
                    self.pool, self._infer, np.concatenate([x for x, _ in items])
                )
            except Exception as e:  # noqa: BLE001 -- propagate to every caller
                for _, f in items:
                    if not f.done():
                        f.set_exception(e)
                continue

            i = 0
            for x, f in items:
                if not f.done():
                    f.set_result(ys[i : i + len(x)])
                i += len(x)


class PredictRequest(BaseModel):
    input: list[list[float]]  # batch of samples, each 150 floats


_batcher: Batcher | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _batcher
    # Runs once per worker process. One forward pass at a time per process, so
    # keep threading explicit to avoid oversubscription across workers.
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(1)
    _batcher = Batcher(load_model(), MAX_BATCH, MAX_QUEUE)
    task = asyncio.create_task(_batcher.run())
    yield
    task.cancel()
    _batcher.pool.shutdown(wait=False)


app = FastAPI(title="FastAPI Batched Inference Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(body: PredictRequest):
    # Validate per request so one malformed payload can't fail a whole batch.
    arr = np.asarray(body.input, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != INPUT_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"input must be a non-empty list of {INPUT_SIZE}-float samples",
        )
    x = arr.reshape(-1, HISTORY_STEPS, HISTORY_FEATURES)

    try:
        out = await asyncio.wait_for(_batcher.submit(x), REQUEST_TIMEOUT_S)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server overloaded")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="inference timed out")

    return {"output": out.reshape(len(out), -1).tolist()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("WORKERS", "1"))
    )
    args = parser.parse_args()
    uvicorn.run(
        "servers.fastapi_batched.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
    )

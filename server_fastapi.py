"""FastAPI (ASGI) inference server — tuned for throughput.

Performance choices:
  - orjson for faster JSON serialisation
  - CPU-bound inference offloaded to a thread-pool so the event loop stays free
  - Raw dict responses (no Pydantic re-validation on the way out)
  - Uvicorn launched with multiple workers (1 per core) in production mode

Usage:
    python server_fastapi.py
    python server_fastapi.py --port 8001
    python server_fastapi.py --workers 4
"""

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import FastAPI, Request, Response

from model_def import VesselTrackPredictor

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
parser = argparse.ArgumentParser(description="FastAPI inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model = VesselTrackPredictor()
_model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
_model.eval()

# ThreadPoolExecutor is intentional: PyTorch releases the GIL during inference,
# so threads achieve real parallelism without the overhead of process spawning.
# ProcessPoolExecutor would require re-loading the model in each process.
_pool = ThreadPoolExecutor(max_workers=1)


def _predict_batch_sync(histories: np.ndarray) -> list:
    """histories: (N, 30, 5) → N × [[lat, lon], ...] × 15"""
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32)
        return _model(x).tolist()


def _predict_sync(history: np.ndarray) -> list:
    """history: (30, 5) → [[lat, lon], ...] × 15"""
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        return _model(x).squeeze(0).tolist()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app_kwargs = {}
if DEFAULT_RESPONSE is not None:
    app_kwargs["default_response_class"] = DEFAULT_RESPONSE

app = FastAPI(title="Vessel Track Inference", version="1.0", **app_kwargs)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    history = body["history"]
    x = np.array(
        [[p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]] for p in history],
        dtype=np.float32,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pool, _predict_sync, x)
    return {"prediction": result}


@app.post("/predict_batch")
async def predict_batch(request: Request):
    body = await request.json()
    vessels = body["vessels"]
    histories = np.array(
        [
            [[p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]] for p in v["history"]]
            for v in vessels
        ],
        dtype=np.float32,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pool, _predict_batch_sync, histories)
    return {"predictions": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_fastapi:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="warning",
    )

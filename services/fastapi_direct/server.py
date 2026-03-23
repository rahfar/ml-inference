"""FastAPI (ASGI) inference server — direct, no queue.

Performance choices:
  - orjson for faster JSON serialisation
  - CPU-bound inference offloaded to a thread-pool so the event loop stays free
  - Raw dict responses (no Pydantic re-validation on the way out)

Usage:
    python services/fastapi_direct/server.py
    python services/fastapi_direct/server.py --port 8001
"""

import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# project root on sys.path so model_def is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
from fastapi import FastAPI, Request, Response

from model_def import VesselTrackPredictor

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "pytorch_model.pt")
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FastAPI direct inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = VesselTrackPredictor().to(_DEVICE)
_model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True, map_location=_DEVICE))
_model.eval()

# PyTorch releases the GIL during inference, so threads achieve real parallelism.
_pool = ThreadPoolExecutor(max_workers=4)

try:
    import orjson

    class ORJSONResponse(Response):
        media_type = "application/json"

        def render(self, content: object) -> bytes:
            return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)

    DEFAULT_RESPONSE = ORJSONResponse
except ImportError:
    DEFAULT_RESPONSE = None


def _predict_batch_sync(histories: np.ndarray) -> list:
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32).to(_DEVICE)
        return _model(x).tolist()


def _predict_sync(history: np.ndarray) -> list:
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
        return _model(x).squeeze(0).tolist()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app_kwargs = {}
if DEFAULT_RESPONSE is not None:
    app_kwargs["default_response_class"] = DEFAULT_RESPONSE

app = FastAPI(title="Vessel Track Inference (Direct)", version="1.0", **app_kwargs)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    history = body["history"]
    x = np.array(
        [
            [p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]]
            for p in history
        ],
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
            [
                [p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]]
                for p in v["history"]
            ]
            for v in vessels
        ],
        dtype=np.float32,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pool, _predict_batch_sync, histories)
    return {"predictions": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

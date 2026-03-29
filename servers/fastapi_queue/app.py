"""FastAPI + async job queue server.

POST /predict        -- submit job, returns {"job_id": "..."}
GET  /result/{id}    -- poll for result
GET  /health         -- readiness check
"""

import asyncio
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))
from model import HISTORY_FEATURES, HISTORY_STEPS, VesselTrackPredictor

WEIGHTS = (
    Path(__file__).resolve().parent.parent.parent / "model" / "weights" / "model.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: VesselTrackPredictor | None = None


def get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor().to(DEVICE)
        _model.load_state_dict(
            torch.load(WEIGHTS, weights_only=True, map_location=DEVICE)
        )
        _model.eval()
    return _model


class PredictRequest(BaseModel):
    input: list[list[float]]  # batch of samples, each 150 floats


# Job storage
_queue: asyncio.Queue | None = None
_results: dict[str, dict] = {}


app = FastAPI(title="FastAPI Queue Inference Server")


@app.on_event("startup")
async def startup():
    global _queue
    get_model()
    _queue = asyncio.Queue()
    asyncio.create_task(_worker())


async def _worker():
    """Background worker that drains the queue and runs inference."""
    while True:
        job_id, data = await _queue.get()
        try:
            batch_size = len(data)
            arr = np.array(data, dtype=np.float32).reshape(
                batch_size, HISTORY_STEPS, HISTORY_FEATURES
            )
            with torch.inference_mode():
                tensor = torch.from_numpy(arr).to(DEVICE)
                out = get_model()(tensor).cpu()
            _results[job_id] = {
                "status": "done",
                "output": [sample.flatten().tolist() for sample in out],
            }
        except Exception as e:
            _results[job_id] = {"status": "error", "detail": str(e)}
        finally:
            _queue.task_done()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(body: PredictRequest):
    job_id = str(uuid.uuid4())
    _results[job_id] = {"status": "pending"}
    await _queue.put((job_id, body.input))
    return {"job_id": job_id}


@app.get("/result/{job_id}")
async def result(job_id: str):
    if job_id not in _results:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _results[job_id]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

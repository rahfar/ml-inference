"""FastAPI inference server -- baseline REST.

POST /predict  -- run inference, return result inline
GET  /health   -- readiness check
"""

import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))
from model import (
    HISTORY_FEATURES,
    HISTORY_STEPS,
    VesselTrackPredictor,
)

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
    input: list[list[float]]  # batch of samples, each 150 floats (30 steps x 5 features)


app = FastAPI(title="FastAPI Inference Server")


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest):
    batch_size = len(body.input)
    arr = np.array(body.input, dtype=np.float32).reshape(
        batch_size, HISTORY_STEPS, HISTORY_FEATURES
    )
    with torch.inference_mode():
        tensor = torch.from_numpy(arr).to(DEVICE)
        out = get_model()(tensor).cpu()
    return {"output": [sample.flatten().tolist() for sample in out]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

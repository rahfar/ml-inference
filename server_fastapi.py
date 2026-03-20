"""FastAPI (ASGI) inference server.

Usage:
    python server_fastapi.py
    python server_fastapi.py --port 8001
"""

import argparse

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model_def import VesselTrackPredictor

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


def _predict(history: np.ndarray) -> list[list[float]]:
    """history: (30, 5) → [[lat, lon], ...] × 15"""
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, 30, 5)
        return _model(x).squeeze(0).tolist()  # (15, 2)


def _predict_batch(histories: np.ndarray) -> list[list[list[float]]]:
    """histories: (N, 30, 5) → N × [[lat, lon], ...] × 15"""
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32)  # (N, 30, 5)
        return _model(x).tolist()  # (N, 15, 2)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vessel Track Inference", version="1.0")


class TrackPoint(BaseModel):
    lat: float
    lon: float
    speed: float
    course_sin: float
    course_cos: float


class PredictRequest(BaseModel):
    history: list[TrackPoint]  # expected length: HISTORY_STEPS


class PredictResponse(BaseModel):
    prediction: list[list[float]]  # 15 × [lat, lon]


class PredictBatchRequest(BaseModel):
    vessels: list[PredictRequest]  # N vessels, each with 30-point history


class PredictBatchResponse(BaseModel):
    predictions: list[list[list[float]]]  # N × 15 × [lat, lon]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(
        [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in req.history],
        dtype=np.float32,
    )
    return PredictResponse(prediction=_predict(x))


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    histories = np.array(
        [
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in v.history]
            for v in req.vessels
        ],
        dtype=np.float32,
    )
    return PredictBatchResponse(predictions=_predict_batch(histories))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

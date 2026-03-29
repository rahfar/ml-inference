import argparse
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

# project root on sys.path so model_def is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VesselTrackPredictor().to(DEVICE)
model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True, map_location=DEVICE))
model.eval()


def _predict_batch_sync(histories: np.ndarray) -> list:
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32).to(DEVICE)
        return model(x).tolist()


def _predict_sync(history: np.ndarray) -> list:
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return model(x).squeeze(0).tolist()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class TrackPoint(BaseModel):
    lat: float
    lon: float
    speed: float
    course_sin: float
    course_cos: float


class PredictRequest(BaseModel):
    history: list[TrackPoint]


class PredictBatchRequest(BaseModel):
    vessels: list[PredictRequest]


app = FastAPI(title="Vessel Track Inference (Direct)", version="1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest):
    x = np.array(
        [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in body.history],
        dtype=np.float32,
    )
    result = _predict_sync(x)
    return {"prediction": result}


@app.post("/predict_batch")
def predict_batch(body: PredictBatchRequest):
    histories = np.array(
        [
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in v.history]
            for v in body.vessels
        ],
        dtype=np.float32,
    )
    result = _predict_batch_sync(histories)
    return {"predictions": result}


if __name__ == "__main__":
    logger.info(f"server starting, listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

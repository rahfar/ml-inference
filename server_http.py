"""FastAPI inference server.

Usage:
    uv run server_http.py
    uv run server_http.py --host 0.0.0.0 --port 8000
"""

import argparse

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import inference

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FastAPI inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Schema
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vessel Track Inference", version="1.0")


@app.on_event("startup")
async def startup():
    inference.get_model()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest):
    history = np.array(
        [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in body.history],
        dtype=np.float32,
    )
    return {"prediction": inference.predict(history)}


@app.post("/predict_batch")
def predict_batch(body: PredictBatchRequest):
    histories = np.array(
        [
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in v.history]
            for v in body.vessels
        ],
        dtype=np.float32,
    )
    return {"predictions": inference.predict_batch(histories)}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

"""Flask (WSGI) inference server served via Gunicorn.

Usage:
    python server_flask.py
    python server_flask.py --port 8001
"""
import numpy as np
import torch
from flask import Flask, jsonify, request

from model_def import VesselTrackPredictor

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
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    history = np.array(
        [[p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]]
         for p in data["history"]],
        dtype=np.float32,
    )
    return jsonify({"prediction": _predict(history)})


@app.post("/predict_batch")
def predict_batch():
    data = request.get_json(force=True)
    histories = np.array(
        [[[p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]]
          for p in vessel["history"]]
         for vessel in data["vessels"]],
        dtype=np.float32,
    )
    return jsonify({"predictions": _predict_batch(histories)})


if __name__ == "__main__":
    import argparse, subprocess, sys

    parser = argparse.ArgumentParser(description="Flask inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Serving Flask+Gunicorn on {args.host}:{args.port}")
    subprocess.run([
        sys.executable, "-m", "gunicorn",
        "server_flask:app",
        "--bind", f"{args.host}:{args.port}",
        "--workers", "1",
        "--threads", "4",
        "--log-level", "warning",
    ])

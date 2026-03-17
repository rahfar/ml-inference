"""Flask (WSGI) inference server served via Waitress.

Usage:
    python server_flask.py
    python server_flask.py --port 8001
"""
import argparse

import numpy as np
import torch
from flask import Flask, jsonify, request

from model_def import VesselTrackPredictor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Flask inference server")
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


if __name__ == "__main__":
    from waitress import serve

    print(f"Serving Flask+Waitress on {args.host}:{args.port}")
    serve(app, host=args.host, port=args.port, threads=4)

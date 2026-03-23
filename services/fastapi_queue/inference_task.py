"""Inference task executed by RQ workers.

The model is loaded lazily on first call and cached for the lifetime of the
worker process — workers fork once, so each process loads the model exactly once.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from model_def import VesselTrackPredictor

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "pytorch_model.pt")
)

_model: VesselTrackPredictor | None = None


def _get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor()
        _model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
        _model.eval()
    return _model


def predict_batch(vessels: list[dict]) -> list[list[list[float]]]:
    """Run batch inference. Called by RQ worker.

    Args:
        vessels: list of {"history": [{lat, lon, speed, course_sin, course_cos} × 30]}

    Returns:
        N × [[lat, lon] × 15]
    """
    model = _get_model()
    histories = np.array(
        [
            [[p["lat"], p["lon"], p["speed"], p["course_sin"], p["course_cos"]] for p in v["history"]]
            for v in vessels
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32)
        return model(x).tolist()

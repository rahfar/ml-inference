"""Shared inference engine — loaded once, used by both HTTP and gRPC servers."""

import os
from pathlib import Path

import numpy as np
import torch

from model_def import VesselTrackPredictor

_MODEL_PATH = Path(__file__).parent / "models" / "pytorch_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: VesselTrackPredictor | None = None


def get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor().to(DEVICE)
        _model.load_state_dict(
            torch.load(_MODEL_PATH, weights_only=True, map_location=DEVICE)
        )
        _model.eval()
    return _model


def predict(history: np.ndarray) -> list[list[float]]:
    """Single vessel. history: (30, 5) float32 array. Returns (15, 2) list."""
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return get_model()(x).squeeze(0).tolist()


def predict_batch(histories: np.ndarray) -> list[list[list[float]]]:
    """Batch of vessels. histories: (N, 30, 5) float32 array. Returns (N, 15, 2) list."""
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32).to(DEVICE)
        return get_model()(x).tolist()

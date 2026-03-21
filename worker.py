"""RQ worker module: prediction function executed by rq workers.

Workers are started with:
    rq worker inference --url redis://localhost:6379

The model is loaded lazily on first invocation and cached for the lifetime
of the worker process.
"""

import numpy as np
import torch

from model_def import VesselTrackPredictor

_model: VesselTrackPredictor | None = None


def _get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor()
        _model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
        _model.eval()
    return _model


def predict_batch(histories: list) -> list:
    """Run batch inference.

    Args:
        histories: N × (30 × 5) nested lists of track points

    Returns:
        N × (15 × 2) nested lists of predicted waypoints
    """
    model = _get_model()
    x = torch.tensor(np.array(histories, dtype=np.float32))
    with torch.no_grad():
        return model(x).tolist()

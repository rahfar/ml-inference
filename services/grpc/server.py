"""gRPC inference server.

Usage:
    python services/grpc/server.py
    python services/grpc/server.py --port 8001
"""

import argparse
import concurrent.futures
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))  # for inference_pb2 stubs

import grpc
import numpy as np
import torch

import inference_pb2
import inference_pb2_grpc
from model_def import VesselTrackPredictor

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "pytorch_model.pt")
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="gRPC inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = VesselTrackPredictor().to(_DEVICE)
_model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True, map_location=_DEVICE))
_model.eval()


def _predict(history: np.ndarray) -> list[list[float]]:
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
        return _model(x).squeeze(0).tolist()


def _predict_batch(histories: np.ndarray) -> list[list[list[float]]]:
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.float32).to(_DEVICE)
        return _model(x).tolist()


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------
class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        history = np.array(
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in request.history],
            dtype=np.float32,
        )
        result = _predict(history)
        return inference_pb2.PredictResponse(
            prediction=[inference_pb2.FuturePoint(lat=pt[0], lon=pt[1]) for pt in result]
        )

    def PredictBatch(self, request, context):
        histories = np.array(
            [[[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in v.history] for v in request.vessels],
            dtype=np.float32,
        )
        results = _predict_batch(histories)
        return inference_pb2.PredictBatchResponse(
            predictions=[
                inference_pb2.PredictResponse(
                    prediction=[inference_pb2.FuturePoint(lat=pt[0], lon=pt[1]) for pt in pred]
                )
                for pred in results
            ]
        )

    def Health(self, request, context):
        return inference_pb2.HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"Serving gRPC on {args.host}:{args.port}")
    server.wait_for_termination()

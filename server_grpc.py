"""gRPC inference server.

Usage:
    python server_grpc.py
    python server_grpc.py --port 8001
"""
import argparse
import concurrent.futures

import grpc
import numpy as np
import torch

import inference_pb2
import inference_pb2_grpc
from model_def import VesselTrackPredictor

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
_model = VesselTrackPredictor()
_model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
_model.eval()


def _predict(history: np.ndarray) -> list[list[float]]:
    """history: (30, 5) → [[lat, lon], ...] × 15"""
    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, 30, 5)
        return _model(x).squeeze(0).tolist()  # (15, 2)


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------
class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        history = np.array(
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos]
             for p in request.history],
            dtype=np.float32,
        )
        result = _predict(history)
        return inference_pb2.PredictResponse(
            prediction=[
                inference_pb2.FuturePoint(lat=pt[0], lon=pt[1]) for pt in result
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

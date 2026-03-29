"""gRPC inference server -- binary protocol over HTTP/2.

Single RPC: Predict(PredictRequest) -> PredictResponse
"""

import concurrent.futures
import sys
import uuid
from pathlib import Path

import grpc
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))

import inference_pb2
import inference_pb2_grpc

from model import HISTORY_FEATURES, HISTORY_STEPS, VesselTrackPredictor

WEIGHTS = (
    Path(__file__).resolve().parent.parent.parent / "model" / "weights" / "model.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: VesselTrackPredictor | None = None


def get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor().to(DEVICE)
        _model.load_state_dict(
            torch.load(WEIGHTS, weights_only=True, map_location=DEVICE)
        )
        _model.eval()
    return _model


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def Predict(self, request, context):
        batch_size = len(request.samples)
        flat = [v for sample in request.samples for v in sample.input]
        arr = np.array(flat, dtype=np.float32).reshape(
            batch_size, HISTORY_STEPS, HISTORY_FEATURES
        )
        with torch.inference_mode():
            tensor = torch.from_numpy(arr).to(DEVICE)
            out = get_model()(tensor).cpu()
        outputs = [
            inference_pb2.PredictOutput(output=sample.flatten().tolist())
            for sample in out
        ]
        return inference_pb2.PredictResponse(
            outputs=outputs,
            request_id=str(uuid.uuid4()),
        )


def serve(host: str = "0.0.0.0", port: int = 50051):
    get_model()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"gRPC server listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(args.host, args.port)

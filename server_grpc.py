"""gRPC inference server.

Usage:
    python server_grpc.py --model catboost
    python server_grpc.py --model pytorch
"""
import argparse
import concurrent.futures

import grpc
import numpy as np

import inference_pb2
import inference_pb2_grpc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="gRPC inference server")
parser.add_argument("--model", choices=["catboost", "pytorch"], required=True)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
if args.model == "catboost":
    from catboost import CatBoostRegressor

    _cb_model = CatBoostRegressor()
    _cb_model.load_model("models/catboost_model.cbm")

    def _predict(x1: float, x2: float, x3: float) -> float:
        X = np.array([[x1, x2, x3]], dtype=np.float32)
        return float(_cb_model.predict(X)[0])

else:
    import torch
    from model_def import MLP

    _pt_model = MLP()
    _pt_model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
    _pt_model.eval()

    def _predict(x1: float, x2: float, x3: float) -> float:
        with torch.no_grad():
            x = torch.tensor([[x1, x2, x3]], dtype=torch.float32)
            return float(_pt_model(x).item())


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------
class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        result = _predict(request.x1, request.x2, request.x3)
        return inference_pb2.PredictResponse(prediction=result)

    def Health(self, request, context):
        return inference_pb2.HealthResponse(status="ok", model=args.model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"Serving gRPC on {args.host}:{args.port} model={args.model}")
    server.wait_for_termination()

"""gRPC inference server.

Usage:
    uv run server_grpc.py
    uv run server_grpc.py --host 0.0.0.0 --port 8000
"""

import argparse
import concurrent.futures

import grpc
import numpy as np

import inference
import inference_pb2
import inference_pb2_grpc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="gRPC inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------
class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        history = np.array(
            [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in request.history],
            dtype=np.float32,
        )
        result = inference.predict(history)
        return inference_pb2.PredictResponse(
            prediction=[inference_pb2.FuturePoint(lat=pt[0], lon=pt[1]) for pt in result]
        )

    def PredictBatch(self, request, context):
        histories = np.array(
            [
                [[p.lat, p.lon, p.speed, p.course_sin, p.course_cos] for p in v.history]
                for v in request.vessels
            ],
            dtype=np.float32,
        )
        results = inference.predict_batch(histories)
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
    inference.get_model()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"Serving gRPC on {args.host}:{args.port}")
    server.wait_for_termination()

"""FastAPI inference server -- MessagePack serialization.

POST /predict  -- msgpack in, msgpack out
GET  /health   -- readiness check (JSON)
"""

import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

from model import HISTORY_FEATURES, HISTORY_STEPS, WEIGHTS_PATH, VesselTrackPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: VesselTrackPredictor | None = None


def get_model() -> VesselTrackPredictor:
    global _model
    if _model is None:
        _model = VesselTrackPredictor().to(DEVICE)
        _model.load_state_dict(
            torch.load(WEIGHTS_PATH, weights_only=True, map_location=DEVICE)
        )
        _model.eval()
    return _model


app = FastAPI(title="FastAPI MessagePack Inference Server")


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = msgpack.unpackb(await request.body(), raw=False)
    samples = body["input"]
    batch_size = len(samples)
    arr = np.array(samples, dtype=np.float32).reshape(
        batch_size, HISTORY_STEPS, HISTORY_FEATURES
    )
    with torch.inference_mode():
        tensor = torch.from_numpy(arr).to(DEVICE)
        out = get_model()(tensor).cpu()
    result = {"output": [sample.flatten().tolist() for sample in out]}
    return Response(
        content=msgpack.packb(result, use_bin_type=True),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")

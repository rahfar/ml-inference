"""FastAPI (ASGI) inference server.

Usage:
    python server_fastapi.py --model catboost
    python server_fastapi.py --model pytorch
"""
import argparse
import sys
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="FastAPI inference server")
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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ML Inference", version="1.0")


class Features(BaseModel):
    x1: float
    x2: float
    x3: float


@app.get("/health")
def health():
    return {"status": "ok", "model": args.model}


@app.post("/predict")
def predict(features: Features):
    result = _predict(features.x1, features.x2, features.x3)
    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

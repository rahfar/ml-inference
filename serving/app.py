"""The only server. Model, strategy and codec are chosen on the command line.

    uv run python -m serving.app --model lstm --strategy batched --codec json

POST /predict   body: (n, 150) float32 in the chosen codec -> (n, 30) float32
GET  /health

Every response reports where the server spent its time, in milliseconds:

    Server-Timing: decode;dur=0.08, wait;dur=0.51, predict;dur=2.90, encode;dur=0.03
    X-Batch-Size: 12        (rows in the predict() call that served the request)

`wait` is the time between decoding and the start of predict(): the queue or
thread pool. It can't see time spent before the handler runs, e.g. while
`inline_loop` blocks the event loop.
"""

import argparse
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response

import models
from models.data import INPUT_SIZE
from serving.codecs import CODECS
from serving.strategies import STRATEGIES


def create_app(model: str, strategy: str, codec: str, threads: int) -> FastAPI:
    predictor = models.load(model, threads)
    runner = STRATEGIES[strategy](predictor.predict)
    wire = CODECS[codec]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runner.start()
        yield
        await runner.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(request: Request) -> Response:
        body = await request.body()
        t0 = time.perf_counter()
        try:
            x = wire.decode(body).reshape(-1, INPUT_SIZE)
        except Exception as e:  # noqa: BLE001 -- any decode failure is a bad request
            return Response(f"cannot decode body: {e}", status_code=400)
        if len(x) == 0:
            return Response("empty input", status_code=400)
        t1 = time.perf_counter()
        result = await runner.predict(x)
        t2 = time.perf_counter()
        out = wire.encode(result.y)
        t3 = time.perf_counter()

        phases = {
            "decode": t1 - t0,
            "wait": (t2 - t1) - result.predict_s,
            "predict": result.predict_s,
            "encode": t3 - t2,
        }
        headers = {
            "Server-Timing": ", ".join(
                f"{k};dur={v * 1e3:.3f}" for k, v in phases.items()
            ),
            "X-Batch-Size": str(result.batch_size),
        }
        return Response(out, media_type=wire.content_type, headers=headers)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Inference server")
    p.add_argument("--model", choices=models.MODELS, default="lstm")
    p.add_argument("--strategy", choices=STRATEGIES, default="batched")
    p.add_argument("--codec", choices=CODECS, default="json")
    p.add_argument("--threads", type=int, default=4, help="threads per predict() call")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    app = create_app(args.model, args.strategy, args.codec, args.threads)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Request handling without any web framework: decode -> predict -> encode, timed.

Settings come from environment variables, so that every server (uvicorn,
granian, gunicorn) can build the app from a factory with no arguments:

    SERVING_MODEL     lstm | catboost | noop
    SERVING_STRATEGY  see serving/strategies.py
    SERVING_CODEC     see serving/codecs.py
    SERVING_THREADS   threads per predict() call

Every reply reports where the server spent its time, in milliseconds:

    Server-Timing: decode;dur=0.08, wait;dur=0.51, predict;dur=2.90, encode;dur=0.03
    X-Batch-Size: 12        (rows in the predict() call that served the request)

`wait` is the time between decoding and the start of predict(): the queue or
thread pool. It can't see time spent before the handler runs, e.g. while
`inline_loop` blocks the event loop, or while a WSGI request waits for a thread.
"""

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np

import models
from models.data import INPUT_SIZE
from serving.codecs import CODECS
from serving.strategies import STRATEGIES, Result, timed_predict


@dataclass(frozen=True)
class Reply:
    status: int
    body: bytes
    content_type: str = "text/plain"
    headers: dict[str, str] | None = None


class Handler:
    def __init__(self, model: str, strategy: str, codec: str, threads: int):
        self.model = models.load(model, threads)
        self.strategy = STRATEGIES[strategy](self.model.predict)
        self.codec = CODECS[codec]

    @classmethod
    def from_env(cls) -> "Handler":
        return cls(
            model=os.environ["SERVING_MODEL"],
            strategy=os.environ["SERVING_STRATEGY"],
            codec=os.environ["SERVING_CODEC"],
            threads=int(os.environ["SERVING_THREADS"]),
        )

    @asynccontextmanager
    async def lifespan(self, _app):
        await self.strategy.start()
        yield
        await self.strategy.stop()

    async def handle(self, body: bytes) -> Reply:
        """For async frameworks: predict() runs according to the strategy."""
        t0 = time.perf_counter()
        x = self._decode(body)
        if isinstance(x, Reply):
            return x
        t1 = time.perf_counter()
        result = await self.strategy.predict(x)
        return self._reply(result, t0, t1, time.perf_counter())

    def handle_sync(self, body: bytes) -> Reply:
        """For WSGI frameworks: predict() runs in the request's own thread."""
        t0 = time.perf_counter()
        x = self._decode(body)
        if isinstance(x, Reply):
            return x
        t1 = time.perf_counter()
        result = timed_predict(self.model.predict, x)
        return self._reply(result, t0, t1, time.perf_counter())

    def _decode(self, body: bytes) -> np.ndarray | Reply:
        try:
            x = self.codec.decode(body).reshape(-1, INPUT_SIZE)
        except Exception as e:  # noqa: BLE001 -- any decode failure is a bad request
            return Reply(400, f"cannot decode body: {e}".encode())
        return x if len(x) else Reply(400, b"empty input")

    def _reply(self, result: Result, t0: float, t1: float, t2: float) -> Reply:
        out = self.codec.encode(result.y)
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
        return Reply(200, out, self.codec.content_type, headers)

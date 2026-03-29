"""Async gRPC load generator. Returns raw (latencies, error_count, elapsed)."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "grpc"))

import grpc.aio
import inference_pb2_grpc


async def run(
    host: str,
    port: int,
    request,
    concurrency: int,
    duration: float,
) -> tuple[list[float], int, float]:
    latencies: list[float] = []
    errors = 0
    start = time.monotonic()

    async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)

        async def worker():
            nonlocal errors
            while time.monotonic() - start < duration:
                t0 = time.monotonic()
                try:
                    await stub.PredictBatch(request)
                    latencies.append(time.monotonic() - t0)
                except Exception:
                    errors += 1

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
        elapsed = time.monotonic() - start

    return latencies, errors, elapsed

"""Async HTTP load generator. Returns raw (latencies, error_count, elapsed)."""

import asyncio
import time

import httpx


async def run(
    url: str,
    payload: dict,
    concurrency: int,
    duration: float,
    request_timeout: float = 30.0,
) -> tuple[list[float], int, float]:
    latencies: list[float] = []
    errors = 0
    start = time.monotonic()

    async def worker():
        nonlocal errors
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            while time.monotonic() - start < duration:
                t0 = time.monotonic()
                try:
                    resp = await client.post(f"{url}/predict_batch", json=payload)
                    resp.raise_for_status()
                    latencies.append(time.monotonic() - t0)
                except Exception:
                    errors += 1

    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    return latencies, errors, time.monotonic() - start

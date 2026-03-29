"""Abstract bench runner and result dataclass."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


@dataclass
class BenchResult:
    bench_name: str
    server_name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    error_rate: float
    concurrency: int
    raw_latencies: list[float] = field(default_factory=list, repr=False)


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def make_result(
    bench_name: str,
    server_name: str,
    concurrency: int,
    latencies: list[float],
    errors: int,
    elapsed: float,
) -> BenchResult:
    """Build a BenchResult from raw latency samples (in seconds)."""
    lat_ms = sorted(t * 1000 for t in latencies)
    total = len(latencies) + errors
    return BenchResult(
        bench_name=bench_name,
        server_name=server_name,
        p50_ms=percentile(lat_ms, 50),
        p95_ms=percentile(lat_ms, 95),
        p99_ms=percentile(lat_ms, 99),
        throughput_rps=len(latencies) / elapsed if elapsed > 0 else 0,
        error_rate=errors / total if total > 0 else 0,
        concurrency=concurrency,
        raw_latencies=lat_ms,
    )


class BaseBench(ABC):
    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.config = config
        self.server_cfg = config["servers"][server_name]

    @abstractmethod
    async def run(self) -> list[BenchResult]: ...

    # -- helpers shared by all benches ----------------------------------------

    def _payload(self) -> list[float]:
        """Build a flat 150-float input payload."""
        return [
            58.0 + i * 0.001
            if j == 0
            else 10.0
            if j == 1
            else 12.0
            if j == 2
            else 0.5
            if j == 3
            else 0.866
            for i in range(30)
            for j in range(5)
        ]

    async def warmup(self, n: int = 10):
        for _ in range(n):
            await self.send_request()

    async def send_request(self) -> float:
        """Send one request and return latency in seconds."""
        protocol = self.server_cfg["protocol"]
        if protocol == "grpc":
            return await self._send_grpc()
        elif self.server_name == "fastapi_queue":
            return await self._send_queue()
        else:
            return await self._send_http()

    async def _send_http(self) -> float:
        url = self.server_cfg["url"]
        async with httpx.AsyncClient(timeout=30) as client:
            t0 = time.monotonic()
            resp = await client.post(f"{url}/predict", json={"input": self._payload()})
            resp.raise_for_status()
            return time.monotonic() - t0

    async def _send_grpc(self) -> float:
        import sys
        from pathlib import Path

        import grpc.aio

        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "servers" / "grpc")
        )
        import inference_pb2  # type: ignore
        import inference_pb2_grpc  # type: ignore

        url = self.server_cfg["url"]
        async with grpc.aio.insecure_channel(url) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            t0 = time.monotonic()
            await stub.Predict(inference_pb2.PredictRequest(input=self._payload()))
            return time.monotonic() - t0

    async def _send_queue(self) -> float:
        """Submit + poll loop for the queue server."""
        url = self.server_cfg["url"]
        async with httpx.AsyncClient(timeout=30) as client:
            t0 = time.monotonic()
            resp = await client.post(f"{url}/predict", json={"input": self._payload()})
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            while True:
                poll = await client.get(f"{url}/result/{job_id}")
                poll.raise_for_status()
                data = poll.json()
                if data["status"] == "done":
                    return time.monotonic() - t0
                if data["status"] == "error":
                    raise RuntimeError(data.get("detail", "inference error"))
                await asyncio.sleep(0.002)

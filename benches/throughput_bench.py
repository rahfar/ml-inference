"""Throughput bench -- max RPS under sustained concurrent load."""

import asyncio
import time

from benches.base import BaseBench, BenchResult, make_result


class ThroughputBench(BaseBench):
    async def run(self) -> list[BenchResult]:
        bench_cfg = self.config["benches"]["throughput"]
        duration = bench_cfg["duration_s"]
        warmup_s = bench_cfg["warmup"]
        concurrency = bench_cfg.get("concurrency", 16)

        # Warmup phase
        await self.warmup(warmup_s)

        latencies: list[float] = []
        errors = 0
        start = time.monotonic()

        async def worker():
            nonlocal errors
            while time.monotonic() - start < duration:
                try:
                    lat = await self.send_request()
                    latencies.append(lat)
                except Exception:
                    errors += 1

        await asyncio.gather(
            *[asyncio.create_task(worker()) for _ in range(concurrency)]
        )
        elapsed = time.monotonic() - start

        return [
            make_result(
                bench_name="throughput",
                server_name=self.server_name,
                concurrency=concurrency,
                latencies=latencies,
                errors=errors,
                elapsed=elapsed,
            )
        ]

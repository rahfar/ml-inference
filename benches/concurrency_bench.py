"""Concurrency bench -- latency vs concurrent users curve."""

import asyncio
import time

from benches.base import BaseBench, BenchResult, make_result


class ConcurrencyBench(BaseBench):
    async def run(self) -> list[BenchResult]:
        bench_cfg = self.config["benches"]["concurrency"]
        levels = bench_cfg["levels"]
        requests_per_level = bench_cfg["requests_per_level"]

        await self.warmup(10)

        results: list[BenchResult] = []
        for level in levels:
            latencies, errors, elapsed = await self._run_level(
                level, requests_per_level
            )
            results.append(
                make_result(
                    bench_name="concurrency",
                    server_name=self.server_name,
                    concurrency=level,
                    latencies=latencies,
                    errors=errors,
                    elapsed=elapsed,
                )
            )
        return results

    async def _run_level(
        self, concurrency: int, total_requests: int
    ) -> tuple[list[float], int, float]:
        latencies: list[float] = []
        errors = 0
        sem = asyncio.Semaphore(concurrency)
        start = time.monotonic()

        async def task():
            nonlocal errors
            async with sem:
                try:
                    lat = await self.send_request()
                    latencies.append(lat)
                except Exception:
                    errors += 1

        await asyncio.gather(
            *[asyncio.create_task(task()) for _ in range(total_requests)]
        )
        elapsed = time.monotonic() - start
        return latencies, errors, elapsed

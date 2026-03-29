"""Latency bench -- single-user baseline P50/P95/P99."""

from benches.base import BaseBench, BenchResult, make_result


class LatencyBench(BaseBench):
    async def run(self) -> list[BenchResult]:
        bench_cfg = self.config["benches"]["latency"]
        n_requests = bench_cfg["n_requests"]
        warmup_n = bench_cfg["warmup"]

        await self.warmup(warmup_n)

        latencies: list[float] = []
        errors = 0
        for _ in range(n_requests):
            try:
                lat = await self.send_request()
                latencies.append(lat)
            except Exception:
                errors += 1

        elapsed = sum(latencies) if latencies else 1.0
        return [
            make_result(
                bench_name="latency",
                server_name=self.server_name,
                concurrency=1,
                latencies=latencies,
                errors=errors,
                elapsed=elapsed,
            )
        ]

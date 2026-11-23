"""Latency bench -- single user, sequential requests, P50/P95/P99."""

from benches.base import BaseBench, BenchResult


class LatencyBench(BaseBench):
    name = "latency"

    def run(self) -> list[BenchResult]:
        cfg = self.config["benches"]["latency"]
        return [self.load(1, cfg["warmup_s"], cfg["duration_s"])]

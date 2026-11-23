"""Throughput bench -- sustained load at a fixed concurrency."""

from benches.base import BaseBench, BenchResult


class ThroughputBench(BaseBench):
    name = "throughput"

    def run(self) -> list[BenchResult]:
        cfg = self.config["benches"]["throughput"]
        return [self.load(cfg["concurrency"], cfg["warmup_s"], cfg["duration_s"])]

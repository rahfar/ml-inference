"""Concurrency bench -- latency vs concurrent users curve.

Each level is an independent run: warmup at that concurrency, then a
fixed-duration measurement window.
"""

from benches.base import BaseBench, BenchResult


class ConcurrencyBench(BaseBench):
    name = "concurrency"

    def run(self) -> list[BenchResult]:
        cfg = self.config["benches"]["concurrency"]
        return [
            self.load(level, cfg["warmup_s"], cfg["duration_s"])
            for level in cfg["levels"]
        ]

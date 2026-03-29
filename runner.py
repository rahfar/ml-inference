"""CLI bench runner -- runs on host, hits Dockerised servers.

Usage:
    python runner.py --bench latency --server fastapi
    python runner.py --bench concurrency --server all
    python runner.py --bench all --server all --output html
"""

import argparse
import asyncio

import yaml

from benches.base import BenchResult
from benches.concurrency_bench import ConcurrencyBench
from benches.latency_bench import LatencyBench
from benches.throughput_bench import ThroughputBench
from report import render

BENCH_CLASSES = {
    "latency": LatencyBench,
    "throughput": ThroughputBench,
    "concurrency": ConcurrencyBench,
}

SERVER_NAMES = ["fastapi", "grpc", "fastapi_queue"]


def parse_args():
    p = argparse.ArgumentParser(description="ML inference benchmark runner")
    p.add_argument(
        "--bench",
        choices=["latency", "throughput", "concurrency", "all"],
        default="latency",
    )
    p.add_argument(
        "--server",
        choices=["fastapi", "grpc", "fastapi_queue", "all"],
        default="fastapi",
    )
    p.add_argument(
        "--output",
        choices=["table", "json", "html"],
        default="table",
    )
    p.add_argument(
        "--concurrency", type=int, nargs="+", help="Override concurrency levels"
    )
    p.add_argument(
        "--duration", type=int, help="Override throughput duration (seconds)"
    )
    p.add_argument(
        "--n-requests",
        type=int,
        dest="n_requests",
        help="Override latency request count",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args) -> dict:
    if args.concurrency:
        config["benches"]["concurrency"]["levels"] = args.concurrency
    if args.duration:
        config["benches"]["throughput"]["duration_s"] = args.duration
    if args.n_requests:
        config["benches"]["latency"]["n_requests"] = args.n_requests
    return config


async def run_all(
    bench_names: list[str], server_names: list[str], config: dict
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for bench_name in bench_names:
        cls = BENCH_CLASSES[bench_name]
        for server_name in server_names:
            print(f"\n{'=' * 60}")
            print(f"  {bench_name} -- {server_name}")
            print(f"{'=' * 60}")
            bench = cls(server_name, config)
            await bench.setup()
            try:
                bench_results = await bench.run()
            finally:
                await bench.teardown()
            results.extend(bench_results)
            for r in bench_results:
                print(
                    f"  c={r.concurrency:>3}  "
                    f"P50={r.p50_ms:>8.2f}ms  P95={r.p95_ms:>8.2f}ms  P99={r.p99_ms:>8.2f}ms  "
                    f"RPS={r.throughput_rps:>8.1f}  err={r.error_rate:.1%}"
                )
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    bench_names = list(BENCH_CLASSES.keys()) if args.bench == "all" else [args.bench]
    server_names = SERVER_NAMES if args.server == "all" else [args.server]

    results = asyncio.run(run_all(bench_names, server_names, config))
    output = render(results, fmt=args.output)
    print(output)


if __name__ == "__main__":
    main()

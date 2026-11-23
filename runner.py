"""CLI bench runner.

Usage:
    python runner.py --bench latency --server fastapi
    python runner.py --bench concurrency --server all --output html --out report.html
    python runner.py --bench all --server all --spawn   # start/stop each server itself

Progress goes to stderr; the rendered report goes to stdout (or --out).
"""

import argparse
import sys
from contextlib import nullcontext

import yaml

from benches.base import CLIENT_CPU_WARN_PCT, BenchResult
from benches.concurrency_bench import ConcurrencyBench
from benches.latency_bench import LatencyBench
from benches.server_proc import spawned
from benches.throughput_bench import ThroughputBench
from report import render

BENCH_CLASSES = {
    "latency": LatencyBench,
    "throughput": ThroughputBench,
    "concurrency": ConcurrencyBench,
}

SERVER_NAMES = [
    "fastapi",
    "grpc",
    "fastapi_queue",
    "fastapi_msgpack",
    "fastapi_batched",
]


def parse_args():
    p = argparse.ArgumentParser(description="ML inference benchmark runner")
    p.add_argument("--bench", choices=[*BENCH_CLASSES, "all"], default="latency")
    p.add_argument("--server", choices=[*SERVER_NAMES, "all"], default="fastapi")
    p.add_argument(
        "--spawn",
        action="store_true",
        help="Start a fresh server process (config `cmd`) for every bench run",
    )
    p.add_argument("--output", choices=["table", "json", "html"], default="table")
    p.add_argument("--out", help="Write the report to this file instead of stdout")
    p.add_argument(
        "--concurrency", type=int, nargs="+", help="Override concurrency levels"
    )
    p.add_argument(
        "--duration", type=float, help="Override measurement duration (all benches)"
    )
    p.add_argument(
        "--client-procs", type=int, dest="client_procs", help="Max client processes"
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
        for bench_cfg in config["benches"].values():
            bench_cfg["duration_s"] = args.duration
    if args.client_procs:
        config.setdefault("client", {})["processes"] = args.client_procs
    return config


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def run_all(
    bench_names: list[str], server_names: list[str], config: dict, spawn: bool
) -> list[BenchResult]:
    results: list[BenchResult] = []
    for bench_name in bench_names:
        for server_name in server_names:
            log(f"\n{'=' * 60}\n  {bench_name} -- {server_name}\n{'=' * 60}")
            bench = BENCH_CLASSES[bench_name](server_name, config)
            server_cfg = config["servers"][server_name]
            with spawned(server_name, server_cfg) if spawn else nullcontext():
                bench_results = bench.run()
            for r in bench_results:
                results.append(r)
                warn = (
                    "  <- client-bound"
                    if r.client_cpu_pct >= CLIENT_CPU_WARN_PCT
                    else ""
                )
                log(
                    f"  c={r.concurrency:>3}  "
                    f"P50={r.p50_ms:>8.2f}ms  P95={r.p95_ms:>8.2f}ms  "
                    f"P99={r.p99_ms:>8.2f}ms  RPS={r.throughput_rps:>8.1f}  "
                    f"n={r.n_samples:>6}  err={r.error_rate:.1%}  "
                    f"client={r.client_procs}p/{r.client_cpu_pct:.0f}%{warn}"
                )
                for err, count in list(r.errors.items())[:3]:
                    log(f"         {count}x {err}")
    return results


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    bench_names = list(BENCH_CLASSES) if args.bench == "all" else [args.bench]
    server_names = SERVER_NAMES if args.server == "all" else [args.server]

    results = run_all(bench_names, server_names, config, args.spawn)
    output = render(results, fmt=args.output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        log(f"\nWrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()

"""Benchmark entrypoint.

Usage:
    uv run benchmarks/run.py --server all
    uv run benchmarks/run.py --server grpc --config benchmarks/config/quick.yaml
    uv run benchmarks/run.py --server http --no-spawn --port 8001
"""

import argparse
import asyncio
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from benchmarks.harness.lifecycle import start_server, stop_server, wait_ready
from benchmarks.harness.monitor import start_monitor
from benchmarks.harness.payloads import grpc_request, http_payload
from benchmarks.metrics import stats as stats_mod
from benchmarks.runners import grpc_runner, http_runner

_RESULTS_DIR = _PROJECT_ROOT / "benchmarks" / "results"

_LABELS = {
    "http": "HTTP (FastAPI)",
    "grpc": "gRPC",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="ML inference benchmark")
    p.add_argument(
        "--server",
        choices=["http", "grpc", "all"],
        default="all",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "benchmarks" / "config" / "default.yaml"),
        help="Path to YAML config file",
    )
    p.add_argument("--duration", type=int, help="Override config duration (seconds)")
    p.add_argument("--concurrency", type=int, help="Override config concurrency")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Override config batch_size")
    p.add_argument("--no-spawn", action="store_true", help="Connect to an already-running server")
    p.add_argument("--no-save", action="store_true", help="Skip saving results to benchmarks/results/")
    p.add_argument("--json", action="store_true", help="Print full JSON results to stdout")
    return p.parse_args()


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_one(server: str, host: str, port: int, cfg: dict, no_spawn: bool) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {_LABELS[server]}")
    print(
        f"  Duration={cfg['duration']}s  Concurrency={cfg['concurrency']}"
        f"  Batch={cfg['batch_size']}"
    )
    print(f"{'=' * 60}")

    handle = None
    if not no_spawn:
        handle = start_server(server, host, port, cfg)

    try:
        print("  Waiting for server...", end=" ", flush=True)
        wait_ready(server, host, port, timeout=cfg["health_timeout"])
        print("ready.")

        if cfg.get("warmup", 0) > 0:
            print(f"  Warming up ({cfg['warmup']}s)...", end=" ", flush=True)
            _drive(server, host, port, cfg, duration_override=cfg["warmup"])
            print("done.")

        monitor = None
        if handle is not None:
            monitor = start_monitor(handle.monitor_pids, interval=cfg["monitor_interval"])

        print("  Running load test...", end=" ", flush=True)
        latencies, errors, elapsed = _drive(server, host, port, cfg)
        print("done.")

        monitor_data = {}
        if monitor is not None:
            monitor.stop()
            for label in handle.monitor_pids:
                monitor_data[label] = monitor.stats_for(label)

    finally:
        if handle is not None:
            stop_server(handle)

    perf = stats_mod.compute(latencies, errors, elapsed, cfg["batch_size"])
    result = {"server": server, "config": cfg, **perf, "processes": monitor_data}
    _print_result(result)
    return result


def _drive(server, host, port, cfg, duration_override=None):
    duration = duration_override if duration_override is not None else cfg["duration"]
    if server == "grpc":
        req = grpc_request(cfg["batch_size"])
        return asyncio.run(grpc_runner.run(host, port, req, cfg["concurrency"], duration))
    else:
        payload = http_payload(cfg["batch_size"])
        return asyncio.run(
            http_runner.run(
                f"http://{host}:{port}", payload, cfg["concurrency"], duration, cfg["request_timeout"]
            )
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _print_result(r: dict):
    proc = r.get("processes", {})
    batch = r["config"]["batch_size"]
    print(
        f"\n  Throughput : {r['rps']:.1f} req/s  ×{batch} = {r['vessels_per_sec']:.0f} vessels/s"
        f"  ({r['requests']} requests, {r['errors']} errors)"
    )
    print(
        f"  Latency    : avg={r['lat_avg_ms']:.2f}ms  p50={r['lat_p50_ms']:.2f}ms"
        f"  p95={r['lat_p95_ms']:.2f}ms  p99={r['lat_p99_ms']:.2f}ms  max={r['lat_max_ms']:.2f}ms"
    )
    if "server" in proc:
        s = proc["server"]
        print(
            f"  Memory     : avg={s['mem_avg_mb']:.1f}MB  max={s['mem_max_mb']:.1f}MB"
            f"  CPU avg={s['cpu_avg_pct']:.1f}%  max={s['cpu_max_pct']:.1f}%"
        )


def _print_summary(results: list[dict]):
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    hdr = f"{'Server':<16} {'req/s':>8} {'vessels/s':>10} {'p50 ms':>8} {'p95 ms':>8}"
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        print(
            f"  {r['server']:<16}"
            f" {r['rps']:>8.1f} {r['vessels_per_sec']:>10.0f}"
            f" {r['lat_p50_ms']:>8.2f} {r['lat_p95_ms']:>8.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _save_results(results: list[dict], servers: list[str]):
    _RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    server_tag = "_".join(servers)
    sha = _git_sha()
    fname = _RESULTS_DIR / f"{ts}_{server_tag}_{sha}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_sha": sha,
        "results": results,
    }
    fname.write_text(json.dumps(payload, indent=2))
    print(f"  Results saved → {fname.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    cfg = _load_config(args.config)

    if args.duration is not None:
        cfg["duration"] = args.duration
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    servers = ["http", "grpc"] if args.server == "all" else [args.server]

    results = []
    for srv in servers:
        results.append(run_one(srv, args.host, args.port, cfg, args.no_spawn))

    if len(results) > 1:
        _print_summary(results)

    if not args.no_save:
        _save_results(results, servers)

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

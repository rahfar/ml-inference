"""Load test: measures RPS, latency percentiles, memory and CPU of the server process.

Each request is a batch of BATCH_SIZE vessels; stats are reported both per-request
and as vessels/sec (RPS × BATCH_SIZE).

Servers:
  fastapi_direct  — FastAPI with thread-pool inference (no queue)
  fastapi_queue   — FastAPI + Redis/RQ worker (requires Redis running)
  grpc            — gRPC with thread-pool
  all             — runs all three in sequence

Usage:
    python load_test.py --server fastapi_direct
    python load_test.py --server fastapi_queue --redis-url redis://localhost:6379
    python load_test.py --server grpc
    python load_test.py --server all --duration 30 --concurrency 20
"""

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "services", "grpc"))

import grpc
import httpx
import inference_pb2
import inference_pb2_grpc
import psutil

from model_def import HISTORY_STEPS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Vessel track inference load test")
parser.add_argument(
    "--server",
    choices=["fastapi_direct", "fastapi_queue", "grpc", "all"],
    default="all",
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--duration", type=int, default=20, help="Test duration in seconds")
parser.add_argument(
    "--concurrency", type=int, default=10, help="Concurrent async workers"
)
parser.add_argument("--redis-url", default="redis://localhost:6379")
parser.add_argument(
    "--no-spawn",
    action="store_true",
    help="Connect to an already-running server instead of spawning one",
)
parser.add_argument(
    "--json",
    action="store_true",
    help="Print result as JSON on the last line (for machine parsing)",
)
args = parser.parse_args()

BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Fixed test payloads
# ---------------------------------------------------------------------------
_TRACK = [
    {
        "lat": 58.0 + i * 0.001,
        "lon": 10.0,
        "speed": 12.0,
        "course_sin": 0.5,
        "course_cos": 0.866,
    }
    for i in range(HISTORY_STEPS)
]
_VESSEL = {"history": _TRACK}
HTTP_PAYLOAD = {"vessels": [_VESSEL] * BATCH_SIZE}

_grpc_single = inference_pb2.PredictRequest(  # type: ignore
    history=[
        inference_pb2.TrackPoint(  # type: ignore
            lat=p["lat"],
            lon=p["lon"],
            speed=p["speed"],
            course_sin=p["course_sin"],
            course_cos=p["course_cos"],
        )
        for p in _TRACK
    ]
)
GRPC_REQUEST = inference_pb2.PredictBatchRequest(vessels=[_grpc_single] * BATCH_SIZE)  # type: ignore


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
@dataclass
class ServerHandle:
    procs: list[subprocess.Popen]  # all processes to kill on teardown
    monitor_pid: int  # PID to sample for CPU/memory stats


_SERVER_SCRIPTS = {
    "fastapi_direct": "services/fastapi_direct/server.py",
    "fastapi_queue": "services/fastapi_queue/server.py",
    "grpc": "services/grpc/server.py",
}


def start_server(server: str, host: str, port: int) -> ServerHandle:
    script = _SERVER_SCRIPTS[server]
    cmd = [sys.executable, script, "--host", host, "--port", str(port)]

    if server == "fastapi_queue":
        cmd += ["--redis-url", args.redis_url]

    srv_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if server == "fastapi_queue":
        worker_cmd = [
            sys.executable,
            "services/fastapi_queue/async_worker.py",
            "--redis-url",
            args.redis_url,
            "--threads", "4",
        ]
        worker_proc = subprocess.Popen(
            worker_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ServerHandle(procs=[srv_proc, worker_proc], monitor_pid=worker_proc.pid)

    return ServerHandle(procs=[srv_proc], monitor_pid=srv_proc.pid)


def stop_server(handle: ServerHandle):
    for proc in handle.procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def wait_ready(url: str, timeout: float = 30.0):
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


def wait_ready_grpc(host: str, port: int, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with grpc.insecure_channel(f"{host}:{port}") as channel:
                stub = inference_pb2_grpc.InferenceStub(channel)
                stub.Health(inference_pb2.HealthRequest(), timeout=1.0)  # type: ignore
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(
        f"gRPC server at {host}:{port} did not become ready within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Process monitor (background thread)
# ---------------------------------------------------------------------------
def monitor_process(
    pid: int, interval: float, stop_event: threading.Event, samples: list
):
    try:
        proc = psutil.Process(pid)
        proc.cpu_percent()  # first call initialises the counter
        time.sleep(interval)
        while not stop_event.is_set():
            try:
                mem_mb = proc.memory_info().rss / 1024 / 1024
                cpu_pct = proc.cpu_percent()
                samples.append((mem_mb, cpu_pct))
            except psutil.NoSuchProcess:
                break
            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass


# ---------------------------------------------------------------------------
# Async load generators
# ---------------------------------------------------------------------------
async def load_http(url: str, concurrency: int, duration: float):
    latencies: list[float] = []
    errors = 0
    start = time.monotonic()

    async def worker():
        nonlocal errors
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() - start < duration:
                t0 = time.monotonic()
                try:
                    resp = await client.post(f"{url}/predict_batch", json=HTTP_PAYLOAD)
                    resp.raise_for_status()
                    latencies.append(time.monotonic() - t0)
                except Exception:
                    errors += 1

    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    return latencies, errors, time.monotonic() - start


async def load_grpc(host: str, port: int, concurrency: int, duration: float):
    import grpc.aio

    latencies: list[float] = []
    errors = 0
    start = time.monotonic()

    async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)

        async def worker():
            nonlocal errors
            while time.monotonic() - start < duration:
                t0 = time.monotonic()
                try:
                    await stub.PredictBatch(GRPC_REQUEST)
                    latencies.append(time.monotonic() - t0)
                except Exception:
                    errors += 1

        await asyncio.gather(
            *[asyncio.create_task(worker()) for _ in range(concurrency)]
        )
        elapsed = time.monotonic() - start

    return latencies, errors, elapsed


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_benchmark(server: str) -> dict:
    host = args.host
    port = args.port
    url = f"http://{host}:{port}"

    label = {
        "fastapi_direct": "FASTAPI DIRECT",
        "fastapi_queue": "FASTAPI + RQ QUEUE",
        "grpc": "gRPC",
    }[server]

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(
        f"  Duration={args.duration}s  Concurrency={args.concurrency}  Batch={BATCH_SIZE}"
    )
    print(f"{'=' * 60}")

    handle = None
    if not args.no_spawn:
        handle = start_server(server, host, port)

    try:
        print("  Waiting for server...", end=" ", flush=True)
        if server == "grpc":
            wait_ready_grpc(host, port)
        else:
            wait_ready(url)
        print("ready.")

        samples: list[tuple[float, float]] = []
        stop_evt = threading.Event()
        mon = None
        if handle is not None:
            mon = threading.Thread(
                target=monitor_process,
                args=(handle.monitor_pid, 0.5, stop_evt, samples),
                daemon=True,
            )
            mon.start()

        print("  Running load test...", end=" ", flush=True)
        if server == "grpc":
            latencies, errors, elapsed = asyncio.run(
                load_grpc(host, port, args.concurrency, args.duration)
            )
        else:
            latencies, errors, elapsed = asyncio.run(
                load_http(url, args.concurrency, args.duration)
            )

        if mon is not None:
            stop_evt.set()
            mon.join(timeout=3)
        print("done.")

    finally:
        if handle is not None:
            stop_server(handle)

    # -----------------------------------------------------------------------
    # Compute stats
    # -----------------------------------------------------------------------
    n = len(latencies)
    rps = n / elapsed if elapsed > 0 else 0

    lat_ms = sorted(lat * 1000 for lat in latencies)
    if lat_ms:
        p50 = statistics.median(lat_ms)
        p95 = lat_ms[int(len(lat_ms) * 0.95)]
        p99 = lat_ms[int(len(lat_ms) * 0.99)]
        avg_lat = statistics.mean(lat_ms)
        max_lat = lat_ms[-1]
    else:
        p50 = p95 = p99 = avg_lat = max_lat = float("nan")

    mem_vals = [s[0] for s in samples]
    cpu_vals = [s[1] for s in samples]
    avg_mem = statistics.mean(mem_vals) if mem_vals else float("nan")
    max_mem = max(mem_vals) if mem_vals else float("nan")
    avg_cpu = statistics.mean(cpu_vals) if cpu_vals else float("nan")
    max_cpu = max(cpu_vals) if cpu_vals else float("nan")

    result = dict(
        server=server,
        batch_size=BATCH_SIZE,
        requests=n,
        errors=errors,
        elapsed=elapsed,
        rps=rps,
        vessels_per_sec=rps * BATCH_SIZE,
        lat_avg_ms=avg_lat,
        lat_p50_ms=p50,
        lat_p95_ms=p95,
        lat_p99_ms=p99,
        lat_max_ms=max_lat,
        mem_avg_mb=avg_mem,
        mem_max_mb=max_mem,
        cpu_avg_pct=avg_cpu,
        cpu_max_pct=max_cpu,
    )
    _print_result(result)
    if args.json:
        import json

        print(json.dumps(result))
    return result


def _print_result(r: dict):
    note = " (worker process)" if r["server"] == "fastapi_queue" else ""
    print(
        f"\n  Throughput : {r['rps']:.1f} req/s  ×{BATCH_SIZE} = {r['vessels_per_sec']:.0f} vessels/s"
        f"  ({r['requests']} requests, {r['errors']} errors)"
    )
    print(
        f"  Latency    : avg={r['lat_avg_ms']:.2f}ms  p50={r['lat_p50_ms']:.2f}ms"
        f"  p95={r['lat_p95_ms']:.2f}ms  p99={r['lat_p99_ms']:.2f}ms  max={r['lat_max_ms']:.2f}ms"
    )
    print(f"  Memory{note} : avg={r['mem_avg_mb']:.1f}MB  max={r['mem_max_mb']:.1f}MB")
    print(f"  CPU{note}    : avg={r['cpu_avg_pct']:.1f}%   max={r['cpu_max_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _print_summary(results: list[dict]):
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    hdr = f"{'Server':<18} {'req/s':>8} {'vessels/s':>10} {'p50 ms':>8} {'p95 ms':>8} {'MemMB':>8} {'CPU%':>7}"
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        print(
            f"  {r['server']:<18}"
            f" {r['rps']:>8.1f} {r['vessels_per_sec']:>10.0f}"
            f" {r['lat_p50_ms']:>8.2f} {r['lat_p95_ms']:>8.2f}"
            f" {r['mem_avg_mb']:>8.1f} {r['cpu_avg_pct']:>7.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    servers = (
        ["fastapi_direct", "fastapi_queue", "grpc"]
        if args.server == "all"
        else [args.server]
    )

    results = []
    for srv in servers:
        results.append(run_benchmark(srv))

    if len(results) > 1:
        _print_summary(results)

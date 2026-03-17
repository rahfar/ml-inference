"""Load test: measures RPS, latency percentiles, memory and CPU of the server process.

Usage:
    python load_test.py --server fastapi
    python load_test.py --server grpc
    python load_test.py --server all    # full benchmark matrix
"""

import argparse
import asyncio
import statistics
import subprocess
import sys
import threading
import time

import grpc
import httpx
import psutil

import inference_pb2
import inference_pb2_grpc
from model_def import HISTORY_STEPS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Vessel track inference load test")
parser.add_argument("--server", choices=["fastapi", "flask", "grpc", "all"], default="all")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--duration", type=int, default=20, help="Test duration in seconds")
parser.add_argument("--concurrency", type=int, default=10, help="Concurrent async workers")
args = parser.parse_args()

# Fixed test payload: 30 track points
_TRACK = [
    {"lat": 58.0 + i * 0.001, "lon": 10.0, "speed": 12.0, "course_sin": 0.5, "course_cos": 0.866}
    for i in range(HISTORY_STEPS)
]
HTTP_PAYLOAD = {"history": _TRACK}
GRPC_REQUEST = inference_pb2.PredictRequest(
    history=[
        inference_pb2.TrackPoint(
            lat=p["lat"], lon=p["lon"], speed=p["speed"],
            course_sin=p["course_sin"], course_cos=p["course_cos"],
        )
        for p in _TRACK
    ]
)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def start_server(server: str, host: str, port: int) -> subprocess.Popen:
    scripts = {"fastapi": "server_fastapi.py", "flask": "server_flask.py", "grpc": "server_grpc.py"}
    cmd = [sys.executable, scripts[server], "--host", host, "--port", str(port)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
                stub.Health(inference_pb2.HealthRequest(), timeout=1.0)
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"gRPC server at {host}:{port} did not become ready within {timeout}s")


def stop_server(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Process monitor (background thread)
# ---------------------------------------------------------------------------
def monitor_process(pid: int, interval: float, stop_event: threading.Event, samples: list):
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() - start < duration:
                t0 = time.monotonic()
                try:
                    resp = await client.post(f"{url}/predict", json=HTTP_PAYLOAD)
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
                    await stub.Predict(GRPC_REQUEST)
                    latencies.append(time.monotonic() - t0)
                except Exception:
                    errors += 1

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
        elapsed = time.monotonic() - start

    return latencies, errors, elapsed


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_benchmark(server: str) -> dict:
    host = args.host
    port = args.port
    url  = f"http://{host}:{port}"

    print(f"\n{'=' * 60}")
    print(f"  Server={server.upper()}")
    print(f"  Duration={args.duration}s  Concurrency={args.concurrency}")
    print(f"{'=' * 60}")

    proc = start_server(server, host, port)
    try:
        print("  Starting server...", end=" ", flush=True)
        if server == "grpc":
            wait_ready_grpc(host, port)
        else:
            wait_ready(url)
        print("ready.")

        samples: list[tuple[float, float]] = []
        stop_evt = threading.Event()
        mon = threading.Thread(
            target=monitor_process, args=(proc.pid, 0.5, stop_evt, samples), daemon=True
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
        stop_evt.set()
        mon.join(timeout=3)
        print("done.")

    finally:
        stop_server(proc)

    # -----------------------------------------------------------------------
    # Compute stats
    # -----------------------------------------------------------------------
    n   = len(latencies)
    rps = n / elapsed if elapsed > 0 else 0

    lat_ms = sorted(l * 1000 for l in latencies)
    if lat_ms:
        p50     = statistics.median(lat_ms)
        p95     = lat_ms[int(len(lat_ms) * 0.95)]
        p99     = lat_ms[int(len(lat_ms) * 0.99)]
        avg_lat = statistics.mean(lat_ms)
        max_lat = lat_ms[-1]
    else:
        p50 = p95 = p99 = avg_lat = max_lat = float("nan")

    mem_vals = [s[0] for s in samples]
    cpu_vals = [s[1] for s in samples]
    avg_mem  = statistics.mean(mem_vals) if mem_vals else float("nan")
    max_mem  = max(mem_vals)             if mem_vals else float("nan")
    avg_cpu  = statistics.mean(cpu_vals) if cpu_vals else float("nan")
    max_cpu  = max(cpu_vals)             if cpu_vals else float("nan")

    result = dict(
        server=server,
        requests=n, errors=errors, elapsed=elapsed, rps=rps,
        lat_avg_ms=avg_lat, lat_p50_ms=p50, lat_p95_ms=p95,
        lat_p99_ms=p99, lat_max_ms=max_lat,
        mem_avg_mb=avg_mem, mem_max_mb=max_mem,
        cpu_avg_pct=avg_cpu, cpu_max_pct=max_cpu,
    )
    _print_result(result)
    return result


def _print_result(r: dict):
    print(f"\n  Throughput : {r['rps']:.1f} req/s  ({r['requests']} requests, {r['errors']} errors)")
    print(f"  Latency    : avg={r['lat_avg_ms']:.2f}ms  p50={r['lat_p50_ms']:.2f}ms  p95={r['lat_p95_ms']:.2f}ms  p99={r['lat_p99_ms']:.2f}ms  max={r['lat_max_ms']:.2f}ms")
    print(f"  Memory     : avg={r['mem_avg_mb']:.1f}MB  max={r['mem_max_mb']:.1f}MB")
    print(f"  CPU        : avg={r['cpu_avg_pct']:.1f}%   max={r['cpu_max_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _print_summary(results: list[dict]):
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    hdr = f"{'Server':<10} {'RPS':>8} {'p50 ms':>8} {'p95 ms':>8} {'MemMB':>8} {'CPU%':>7}"
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        print(
            f"  {r['server']:<10}"
            f" {r['rps']:>8.1f} {r['lat_p50_ms']:>8.2f}"
            f" {r['lat_p95_ms']:>8.2f} {r['mem_avg_mb']:>8.1f}"
            f" {r['cpu_avg_pct']:>7.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    servers = ["fastapi", "flask", "grpc"] if args.server == "all" else [args.server]

    results = []
    for srv in servers:
        results.append(run_benchmark(srv))

    if len(results) > 1:
        _print_summary(results)

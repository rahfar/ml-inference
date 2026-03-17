"""Docker-based benchmark runner.

Builds a single image, then for every (server, cpu_limit) combination:
  - starts a container with --cpus=N
  - collects resource usage via `docker stats` in a background thread
  - drives load via load_test.py --no-spawn --json
  - stops the container and records the combined result

Usage:
    uv run bench_docker.py                        # all servers, 1/2/4 vCPUs
    uv run bench_docker.py --server fastapi
    uv run bench_docker.py --cpus 1 2             # only 1 and 2 vCPUs
    uv run bench_docker.py --no-build             # skip docker build
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

IMAGE = "ml-inference-bench"
HTTP_PORT = 8000
GRPC_PORT = 50051


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def build_image():
    print("Building Docker image (this may take a few minutes on first run)...")
    subprocess.run(["docker", "build", "-t", IMAGE, "."], check=True)
    print("Image ready.\n")


def start_container(server: str, cpus: int, mem: str) -> str:
    port = GRPC_PORT if server == "grpc" else HTTP_PORT
    cmd = [
        "docker", "run", "-d", "--rm",
        f"--cpus={cpus}",
        f"--memory={mem}",
        "-p", f"{port}:{port}",
        IMAGE,
        "python", f"server_{server}.py",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def stop_container(container_id: str):
    subprocess.run(["docker", "stop", container_id], capture_output=True)


def _parse_mem_mb(usage: str) -> float:
    """Parse the first part of docker stats MemUsage (e.g. '145.3MiB / 2GiB') to MB."""
    part = usage.split("/")[0].strip()
    if "GiB" in part:
        return float(part.replace("GiB", "").strip()) * 1024
    if "MiB" in part:
        return float(part.replace("MiB", "").strip())
    if "kB" in part or "KiB" in part:
        return float(part.replace("kB", "").replace("KiB", "").strip()) / 1024
    if "MB" in part:
        return float(part.replace("MB", "").strip())
    return float(part)


def collect_docker_stats(container_id: str, stop_event: threading.Event, samples: list):
    fmt = '{"cpu":"{{.CPUPerc}}","mem":"{{.MemUsage}}"}'
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", fmt, container_id],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    cpu_pct = float(data["cpu"].rstrip("%"))
                    mem_mb  = _parse_mem_mb(data["mem"])
                    samples.append((cpu_pct, mem_mb))
        except Exception:
            pass
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Readiness probes
# ---------------------------------------------------------------------------

def wait_ready_http(port: int, timeout: float = 60.0):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"HTTP server on :{port} not ready after {timeout}s")


def wait_ready_grpc(port: int, timeout: float = 60.0):
    import grpc
    import inference_pb2
    import inference_pb2_grpc
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with grpc.insecure_channel(f"localhost:{port}") as ch:
                stub = inference_pb2_grpc.InferenceStub(ch)
                stub.Health(inference_pb2.HealthRequest(), timeout=1.0)
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"gRPC server on :{port} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Load test runner (delegates to load_test.py --no-spawn --json)
# ---------------------------------------------------------------------------

def run_load_test(server: str, port: int, duration: int, concurrency: int) -> dict:
    cmd = [
        sys.executable, "load_test.py",
        "--server", server,
        "--host", "localhost",
        "--port", str(port),
        "--duration", str(duration),
        "--concurrency", str(concurrency),
        "--no-spawn",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # load_test.py prints JSON as the last non-empty line when --json is set
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        f"No JSON output from load_test.py.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(server: str, cpus: int, mem: str, duration: int, concurrency: int) -> dict:
    port  = GRPC_PORT if server == "grpc" else HTTP_PORT
    label = f"{server.upper()}  {cpus} vCPU  {mem} mem"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Duration={duration}s  Concurrency={concurrency}")
    print(f"{'=' * 60}")

    container_id = start_container(server, cpus, mem)
    try:
        print("  Waiting for server...", end=" ", flush=True)
        if server == "grpc":
            wait_ready_grpc(port)
        else:
            wait_ready_http(port)
        print("ready.")

        samples: list[tuple[float, float]] = []
        stop_evt = threading.Event()
        mon = threading.Thread(
            target=collect_docker_stats,
            args=(container_id, stop_evt, samples),
            daemon=True,
        )
        mon.start()

        print("  Running load test...", end=" ", flush=True)
        stats = run_load_test(server, port, duration, concurrency)
        stop_evt.set()
        mon.join(timeout=5)
        print("done.")

    finally:
        stop_container(container_id)

    cpu_vals = [s[0] for s in samples]
    mem_vals = [s[1] for s in samples]

    stats["cpu_limit"]        = cpus
    stats["mem_limit"]        = mem
    stats["docker_cpu_avg"]   = statistics.mean(cpu_vals) if cpu_vals else float("nan")
    stats["docker_cpu_max"]   = max(cpu_vals)             if cpu_vals else float("nan")
    stats["docker_mem_avg_mb"] = statistics.mean(mem_vals) if mem_vals else float("nan")
    stats["docker_mem_max_mb"] = max(mem_vals)             if mem_vals else float("nan")

    _print_result(stats)
    return stats


def _print_result(r: dict):
    bs = r.get("batch_size", 100)
    print(f"\n  Throughput : {r['rps']:.1f} req/s  ×{bs} = {r['vessels_per_sec']:.0f} vessels/s"
          f"  ({r['requests']} requests, {r['errors']} errors)")
    print(f"  Latency    : avg={r['lat_avg_ms']:.1f}ms  p50={r['lat_p50_ms']:.1f}ms"
          f"  p95={r['lat_p95_ms']:.1f}ms  p99={r['lat_p99_ms']:.1f}ms")
    print(f"  Container  : CPU avg={r['docker_cpu_avg']:.1f}%  max={r['docker_cpu_max']:.1f}%"
          f"  |  Mem avg={r['docker_mem_avg_mb']:.0f}MB  max={r['docker_mem_max_mb']:.0f}MB")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]):
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    hdr = (f"{'Server':<10} {'vCPU':>5} {'req/s':>8} {'vessels/s':>10}"
           f" {'p50ms':>7} {'p95ms':>7} {'MemMB':>7} {'CPU%':>7}")
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for r in results:
        print(
            f"  {r['server']:<10}"
            f" {r['cpu_limit']:>5}"
            f" {r['rps']:>8.1f}"
            f" {r['vessels_per_sec']:>10.0f}"
            f" {r['lat_p50_ms']:>7.1f}"
            f" {r['lat_p95_ms']:>7.1f}"
            f" {r['docker_mem_avg_mb']:>7.0f}"
            f" {r['docker_cpu_avg']:>7.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker-based multi-CPU benchmark")
    parser.add_argument("--server", choices=["fastapi", "flask", "grpc", "all"], default="all")
    parser.add_argument("--cpus", nargs="+", type=int, default=[1, 2, 4],
                        help="vCPU limits to test (default: 1 2 4)")
    parser.add_argument("--mem", default="2g",
                        help="Memory limit per container (default: 2g)")
    parser.add_argument("--duration", type=int, default=20,
                        help="Load test duration per run in seconds (default: 20)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent async workers (default: 10)")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip docker build (use existing image)")
    args = parser.parse_args()

    if not args.no_build:
        build_image()

    servers = ["fastapi", "flask", "grpc"] if args.server == "all" else [args.server]

    results = []
    for server in servers:
        for cpus in args.cpus:
            results.append(
                run_benchmark(server, cpus, args.mem, args.duration, args.concurrency)
            )

    _print_summary(results)

"""Server spawn/teardown and health polling."""

import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import grpc

_PROJECT_ROOT = Path(__file__).parent.parent.parent

_SERVER_SCRIPTS = {
    "fastapi_direct": _PROJECT_ROOT / "services/fastapi_direct/server.py",
    "fastapi_queue": _PROJECT_ROOT / "services/fastapi_queue/server.py",
    "grpc": _PROJECT_ROOT / "services/grpc/server.py",
}

_WORKER_SCRIPT = _PROJECT_ROOT / "services/fastapi_queue/async_worker.py"


@dataclass
class ServerHandle:
    server: str
    procs: list[subprocess.Popen] = field(default_factory=list)
    # PIDs to monitor: {label: pid}
    monitor_pids: dict[str, int] = field(default_factory=dict)


def start_server(server: str, host: str, port: int, cfg: dict) -> ServerHandle:
    script = _SERVER_SCRIPTS[server]
    cmd = [sys.executable, str(script), "--host", host, "--port", str(port)]

    if server == "fastapi_queue":
        cmd += ["--redis-url", cfg["redis_url"]]

    srv_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    handle = ServerHandle(
        server=server,
        procs=[srv_proc],
        monitor_pids={"server": srv_proc.pid},
    )

    if server == "fastapi_queue":
        worker_cmd = [
            sys.executable,
            str(_WORKER_SCRIPT),
            "--redis-url", cfg["redis_url"],
            "--threads", str(cfg["worker_threads"]),
        ]
        worker_proc = subprocess.Popen(
            worker_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        handle.procs.append(worker_proc)
        handle.monitor_pids["worker"] = worker_proc.pid

    return handle


def stop_server(handle: ServerHandle):
    for proc in handle.procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def wait_ready_http(url: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


def wait_ready_grpc(host: str, port: int, timeout: float = 30.0):
    sys.path.insert(0, str(_PROJECT_ROOT / "services" / "grpc"))
    import inference_pb2
    import inference_pb2_grpc

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with grpc.insecure_channel(f"{host}:{port}") as channel:
                stub = inference_pb2_grpc.InferenceStub(channel)
                stub.Health(inference_pb2.HealthRequest(), timeout=1.0)
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(
        f"gRPC server at {host}:{port} did not become ready within {timeout}s"
    )


def wait_ready(server: str, host: str, port: int, timeout: float = 30.0):
    if server == "grpc":
        wait_ready_grpc(host, port, timeout)
    else:
        wait_ready_http(f"http://{host}:{port}", timeout)

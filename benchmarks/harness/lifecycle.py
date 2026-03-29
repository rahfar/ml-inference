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
    "http": _PROJECT_ROOT / "server_http.py",
    "grpc": _PROJECT_ROOT / "server_grpc.py",
}


@dataclass
class ServerHandle:
    server: str
    procs: list[subprocess.Popen] = field(default_factory=list)
    monitor_pids: dict[str, int] = field(default_factory=dict)


def start_server(server: str, host: str, port: int, cfg: dict) -> ServerHandle:
    script = _SERVER_SCRIPTS[server]
    cmd = [sys.executable, str(script), "--host", host, "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ServerHandle(server=server, procs=[proc], monitor_pids={"server": proc.pid})


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
    sys.path.insert(0, str(_PROJECT_ROOT))
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

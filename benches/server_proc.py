"""Start a server as a local subprocess for the duration of a bench run.

Each (bench, server) pair gets a fresh process, so no state (caches, leaked
job results, warmed thread pools) carries over between runs. Server output
goes to results/logs/<server>.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOG_DIR = Path("results/logs")


def is_ready(server_cfg: dict) -> bool:
    url = server_cfg["url"]
    if server_cfg["protocol"] == "grpc":
        import grpc

        with grpc.insecure_channel(url) as ch:
            try:
                grpc.channel_ready_future(ch).result(timeout=0.5)
                return True
            except grpc.FutureTimeoutError:
                return False
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=0.5) as resp:
            return resp.status == 200
    except OSError:
        return False


@contextmanager
def spawned(
    server_name: str, server_cfg: dict, timeout_s: float = 60
) -> Iterator[None]:
    if is_ready(server_cfg):
        raise RuntimeError(
            f"{server_name}: something is already serving {server_cfg['url']}; "
            "stop it or run without --spawn"
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{server_name}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, *server_cfg["cmd"]], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + timeout_s
            while not is_ready(server_cfg):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{server_name}: exited with {proc.returncode}, see {log_path}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{server_name}: not ready after {timeout_s}s")
                time.sleep(0.25)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

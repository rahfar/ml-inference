"""Load-generation core shared by all benches.

Methodology:
- Closed-loop load: C virtual users, each sends its next request as soon as the
  previous one returns.
- Users are spread over P client *processes* (spawned, each with its own event
  loop and connections), so a single Python event loop is not the bottleneck.
  Each process reports its CPU utilisation during the measurement window; if any
  process is near 100%, the numbers are client-bound, not server-bound.
- Every run is a warmup window at the target concurrency (discarded; it also
  opens all connections) followed by a fixed-duration measurement window. Only
  requests that *start* inside the window are counted.
- Every protocol is timed the same way: the request body is pre-encoded once,
  the timer covers send -> full response received -> response decoded.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

import aiohttp
import msgpack

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class BenchResult:
    bench_name: str
    server_name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    error_rate: float
    concurrency: int
    n_samples: int = 0
    client_procs: int = 1
    client_cpu_pct: float = 0.0  # busiest client process, % of one core
    errors: dict[str, int] = field(default_factory=dict)
    raw_latencies: list[float] = field(default_factory=list, repr=False)


CLIENT_CPU_WARN_PCT = 80.0


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# --------------------------------------------------------------------------- #
# Protocol clients
# --------------------------------------------------------------------------- #


def sample() -> list[float]:
    """One flat 150-float sample (30 steps x 5 features)."""
    return [v for i in range(30) for v in (58.0 + i * 0.001, 10.0, 12.0, 0.5, 0.866)]


class Client(ABC):
    def __init__(self, url: str, connections: int, payload: list[list[float]]):
        self.url = url
        self.connections = connections
        self.payload = payload

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def request(self) -> None:
        """Send one request, wait for and decode the full response."""

    @abstractmethod
    async def close(self) -> None: ...


class _AiohttpClient(Client):
    session: aiohttp.ClientSession

    async def open(self) -> None:
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.connections),
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def close(self) -> None:
        await self.session.close()


class HttpJsonClient(_AiohttpClient):
    async def open(self) -> None:
        await super().open()
        self.body = json.dumps({"input": self.payload}).encode()
        self.headers = {"Content-Type": "application/json"}

    async def request(self) -> None:
        async with self.session.post(
            f"{self.url}/predict", data=self.body, headers=self.headers
        ) as resp:
            data = await resp.read()
            resp.raise_for_status()
        json.loads(data)


class MsgpackClient(_AiohttpClient):
    async def open(self) -> None:
        await super().open()
        self.body = msgpack.packb({"input": self.payload}, use_bin_type=True)
        self.headers = {"Content-Type": "application/x-msgpack"}

    async def request(self) -> None:
        async with self.session.post(
            f"{self.url}/predict", data=self.body, headers=self.headers
        ) as resp:
            data = await resp.read()
            resp.raise_for_status()
        msgpack.unpackb(data, raw=False)


class QueueClient(HttpJsonClient):
    """Submit + poll pattern of the fastapi_queue server."""

    poll_interval_s = 0.002

    async def request(self) -> None:
        async with self.session.post(
            f"{self.url}/predict", data=self.body, headers=self.headers
        ) as resp:
            data = await resp.read()
            resp.raise_for_status()
        job_id = json.loads(data)["job_id"]
        while True:
            async with self.session.get(f"{self.url}/result/{job_id}") as resp:
                data = await resp.read()
                resp.raise_for_status()
            result = json.loads(data)
            if result["status"] == "done":
                return
            if result["status"] == "error":
                raise RuntimeError(result.get("detail", "inference error"))
            await asyncio.sleep(self.poll_interval_s)


class GrpcClient(Client):
    async def open(self) -> None:
        import grpc.aio

        from servers.grpc import inference_pb2

        self.channel = grpc.aio.insecure_channel(self.url)
        await self.channel.channel_ready()
        # Pre-encoded request bytes (serializer=None passes bytes through).
        self.body = inference_pb2.PredictRequest(
            samples=[inference_pb2.Sample(input=s) for s in self.payload]
        ).SerializeToString()
        self.call = self.channel.unary_unary(
            "/inference.InferenceService/Predict",
            request_serializer=None,
            response_deserializer=inference_pb2.PredictResponse.FromString,
        )

    async def request(self) -> None:
        await self.call(self.body, timeout=30)

    async def close(self) -> None:
        await self.channel.close()


def make_client(server_cfg: dict, url: str, connections: int) -> Client:
    payload = [sample()]
    pattern = server_cfg.get("pattern")
    protocol = server_cfg["protocol"]
    if protocol == "grpc":
        return GrpcClient(url, connections, payload)
    if protocol == "msgpack":
        return MsgpackClient(url, connections, payload)
    if pattern == "queue":
        return QueueClient(url, connections, payload)
    return HttpJsonClient(url, connections, payload)


# --------------------------------------------------------------------------- #
# Multi-process load runner
# --------------------------------------------------------------------------- #


@dataclass
class _ProcResult:
    latencies: list[float]
    errors: Counter
    cpu_pct: float


async def _proc_async(
    server_cfg: dict,
    url: str,
    users: int,
    warmup_s: float,
    duration_s: float,
    barrier,
) -> _ProcResult:
    client = make_client(server_cfg, url, connections=users)
    await client.open()
    try:
        # Fail fast if the server is down, before syncing with other processes.
        await client.request()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, barrier.wait, 120)

        t0 = time.perf_counter()
        m0, m1 = t0 + warmup_s, t0 + warmup_s + duration_s
        latencies: list[float] = []
        errors: Counter = Counter()
        cpu: dict[str, float] = {}

        async def user():
            while (start := time.perf_counter()) < m1:
                try:
                    await client.request()
                except Exception as e:  # noqa: BLE001 -- count every failure
                    if start >= m0:
                        errors[f"{type(e).__name__}: {str(e)[:120]}"] += 1
                    continue
                if start >= m0:
                    latencies.append(time.perf_counter() - start)

        async def cpu_meter():
            await asyncio.sleep(max(0.0, m0 - time.perf_counter()))
            cpu["start"], cpu["t0"] = time.process_time(), time.perf_counter()
            await asyncio.sleep(max(0.0, m1 - time.perf_counter()))
            cpu["end"], cpu["t1"] = time.process_time(), time.perf_counter()

        await asyncio.gather(cpu_meter(), *(user() for _ in range(users)))
        cpu_pct = 100 * (cpu["end"] - cpu["start"]) / (cpu["t1"] - cpu["t0"])
        return _ProcResult(latencies, errors, cpu_pct)
    finally:
        await client.close()


def _proc_main(server_cfg, url, users, warmup_s, duration_s, barrier, out_q):
    try:
        try:
            import uvloop

            uvloop.install()
        except ImportError:
            pass
        out_q.put(
            asyncio.run(
                _proc_async(server_cfg, url, users, warmup_s, duration_s, barrier)
            )
        )
    except BaseException as e:  # noqa: BLE001 -- report to parent, unblock peers
        barrier.abort()
        out_q.put(e if isinstance(e, Exception) else RuntimeError(repr(e)))


def run_load(
    bench_name: str,
    server_name: str,
    server_cfg: dict,
    url: str,
    concurrency: int,
    warmup_s: float,
    duration_s: float,
    max_procs: int,
) -> BenchResult:
    procs = max(1, min(max_procs, concurrency))
    shares = [concurrency // procs + (i < concurrency % procs) for i in range(procs)]

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(procs)
    out_q = ctx.Queue()
    workers = [
        ctx.Process(
            target=_proc_main,
            args=(server_cfg, url, n, warmup_s, duration_s, barrier, out_q),
            daemon=True,
        )
        for n in shares
    ]
    for w in workers:
        w.start()
    results = [out_q.get(timeout=warmup_s + duration_s + 180) for _ in workers]
    for w in workers:
        w.join()
    for r in results:
        if isinstance(r, Exception):
            raise RuntimeError(  # noqa: TRY004
                f"{server_name}: client process failed: {r!r}"
            ) from r

    lat_ms = sorted(x * 1000 for r in results for x in r.latencies)
    errors: Counter = sum((r.errors for r in results), Counter())
    n_err = sum(errors.values())
    total = len(lat_ms) + n_err
    return BenchResult(
        bench_name=bench_name,
        server_name=server_name,
        p50_ms=percentile(lat_ms, 50),
        p95_ms=percentile(lat_ms, 95),
        p99_ms=percentile(lat_ms, 99),
        throughput_rps=len(lat_ms) / duration_s,
        error_rate=n_err / total if total else 0.0,
        concurrency=concurrency,
        n_samples=len(lat_ms),
        client_procs=procs,
        client_cpu_pct=max(r.cpu_pct for r in results),
        errors=dict(errors),
        raw_latencies=lat_ms,
    )


# --------------------------------------------------------------------------- #
# Bench base class
# --------------------------------------------------------------------------- #


class BaseBench(ABC):
    def __init__(self, server_name: str, config: dict, target: str = "host"):
        self.server_name = server_name
        self.config = config
        self.server_cfg = config["servers"][server_name]
        self.url = self.server_cfg["docker_url" if target == "docker" else "url"]
        self.max_procs = config.get("client", {}).get("processes", 4)

    @abstractmethod
    def run(self) -> list[BenchResult]: ...

    def load(self, concurrency: int, warmup_s: float, duration_s: float) -> BenchResult:
        return run_load(
            bench_name=self.name,
            server_name=self.server_name,
            server_cfg=self.server_cfg,
            url=self.url,
            concurrency=concurrency,
            warmup_s=warmup_s,
            duration_s=duration_s,
            max_procs=self.max_procs,
        )

    name: str = ""

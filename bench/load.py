"""Closed-loop load generator for one concurrency level.

C virtual users each send their next request as soon as the previous one
returns. They are spread over up to P client processes, so a single Python
event loop doesn't become the bottleneck. Each level is a warmup window
(discarded; it also opens the connections) followed by a fixed measurement
window, and only requests that start inside that window count.

Every process reports its CPU use. If the busiest one is near 100% of a core,
the result measures the client rather than the server.
"""

import asyncio
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import dataclass

import aiohttp
import numpy as np

from models.data import make_dataset
from serving.codecs import CODECS

CLIENT_CPU_WARN_PCT = 80.0


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    rps: float
    p50_ms: float
    p99_ms: float
    errors: int
    client_cpu_pct: float  # busiest client process, % of one core
    server_ms: dict[str, float]  # mean per request: decode / wait / predict / encode
    batch_size: float  # mean rows in the predict() call that served a request


@dataclass
class _ProcessResult:
    latencies: list[float]
    server_sums: Counter  # sums of the server-reported values
    errors: Counter
    cpu_pct: float


def _parse_server_timing(header: str) -> dict[str, float]:
    """'decode;dur=0.1, wait;dur=0.4' -> {'decode': 0.1, 'wait': 0.4}"""
    phases = {}
    for part in header.split(","):
        name, _, duration = part.strip().partition(";dur=")
        if duration:
            phases[name] = float(duration)
    return phases


async def _client(url, codec_name, samples, users, warmup_s, duration_s, barrier):
    codec = CODECS[codec_name]
    body = codec.encode(make_dataset(samples, seed=7)[0])
    headers = {"Content-Type": codec.content_type}
    latencies: list[float] = []
    server_sums: Counter = Counter()
    errors: Counter = Counter()
    cpu: dict[str, float] = {}

    connector = aiohttp.TCPConnector(limit=users)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def request() -> dict[str, float]:
            async with session.post(
                f"{url}/predict", data=body, headers=headers
            ) as resp:
                data = await resp.read()
                resp.raise_for_status()
                reported = _parse_server_timing(resp.headers.get("Server-Timing", ""))
                reported["batch_size"] = float(resp.headers.get("X-Batch-Size", 0))
            codec.decode(data)
            return reported

        await request()  # fail fast if the server is down, before syncing
        await asyncio.get_running_loop().run_in_executor(None, barrier.wait, 120)
        t0 = time.perf_counter()
        m0, m1 = t0 + warmup_s, t0 + warmup_s + duration_s

        async def user() -> None:
            while (start := time.perf_counter()) < m1:
                try:
                    reported = await request()
                except Exception as e:  # noqa: BLE001 -- count every failure
                    if start >= m0:
                        errors[f"{type(e).__name__}: {e}"[:150]] += 1
                    continue
                if start >= m0:
                    latencies.append(time.perf_counter() - start)
                    server_sums.update(reported)

        async def cpu_meter() -> None:
            await asyncio.sleep(m0 - time.perf_counter())
            cpu["start"], cpu["t0"] = time.process_time(), time.perf_counter()
            await asyncio.sleep(m1 - time.perf_counter())
            cpu["end"], cpu["t1"] = time.process_time(), time.perf_counter()

        await asyncio.gather(cpu_meter(), *(user() for _ in range(users)))

    cpu_pct = 100 * (cpu["end"] - cpu["start"]) / (cpu["t1"] - cpu["t0"])
    return _ProcessResult(latencies, server_sums, errors, cpu_pct)


def _process_main(args, barrier, out_queue) -> None:
    try:
        import uvloop

        uvloop.install()
        out_queue.put(asyncio.run(_client(*args, barrier)))
    except BaseException as e:  # noqa: BLE001 -- report to the parent, free the peers
        barrier.abort()
        out_queue.put(RuntimeError(repr(e)))


def run_level(
    url: str,
    codec: str,
    samples: int,
    concurrency: int,
    warmup_s: float,
    duration_s: float,
    max_processes: int,
) -> LevelResult:
    n_proc = max(1, min(max_processes, concurrency))
    users = [concurrency // n_proc + (i < concurrency % n_proc) for i in range(n_proc)]

    ctx = mp.get_context("spawn")
    barrier, out_queue = ctx.Barrier(n_proc), ctx.Queue()
    processes = [
        ctx.Process(
            target=_process_main,
            args=((url, codec, samples, n, warmup_s, duration_s), barrier, out_queue),
            daemon=True,
        )
        for n in users
    ]
    for p in processes:
        p.start()
    results = [out_queue.get(timeout=warmup_s + duration_s + 180) for _ in processes]
    for p in processes:
        p.join()
    for r in results:
        if isinstance(r, Exception):
            raise RuntimeError(f"client process failed: {r}")  # noqa: TRY004

    latencies_ms = np.array([x for r in results for x in r.latencies]) * 1e3
    n = len(latencies_ms)
    sums = sum((r.server_sums for r in results), Counter())
    p50, p99 = np.percentile(latencies_ms, [50, 99]) if n else (float("nan"),) * 2
    return LevelResult(
        concurrency=concurrency,
        requests=n,
        rps=n / duration_s,
        p50_ms=float(p50),
        p99_ms=float(p99),
        errors=sum(sum(r.errors.values()) for r in results),
        client_cpu_pct=max(r.cpu_pct for r in results),
        server_ms={k: sums[k] / n for k in ("decode", "wait", "predict", "encode")}
        if n
        else {},
        batch_size=sums["batch_size"] / n if n else 0.0,
    )

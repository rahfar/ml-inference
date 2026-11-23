# ML Inference Bench

Benchmark and compare PyTorch serving strategies for a vessel track predictor LSTM model (~3 MB, hidden=256, layers=2).

## Servers

| Server | Protocol | Port | Pattern |
|---|---|---|---|
| `fastapi` | HTTP/REST | 8001 | Sync inference in async handler |
| `grpc` | gRPC/HTTP2 | 8002 | Binary protocol, thread pool (4 workers) |
| `fastapi_queue` | HTTP/REST | 8003 | Submit → poll via `asyncio.Queue` |
| `fastapi_msgpack` | HTTP/MessagePack | 8004 | Inline inference, msgpack body |
| `go` | HTTP/REST | 8005 | Go + ONNX Runtime |
| `fastapi_batched` | HTTP/REST | 8006 | In-process dynamic batching, 1 inference thread per process (torch) |

**Model:** LSTM encoder → linear decoder
- Input: 30 steps x 5 features (150 floats)
- Output: 15 steps x 2 features (30 floats)

## Methodology

`make bench-all` runs every bench against every server in isolation:

- **Fresh container per server.** It's started, benchmarked and torn down before the next server, so no state carries over between runs.
- **Client and server on separate cores.** Servers are pinned to `cpuset: 0-3` (`SERVER_CPUS`). The load generator runs in its own container on `cpuset: 4-9` (`CLIENT_CPUS`), on the compose network, so there's no Docker Desktop port forwarding.
- **Multi-process client.** Up to 4 spawned processes (aiohttp + uvloop, or grpc.aio), each with its own event loop and connections. Every result reports the CPU use of the busiest client process. At or above 80% the run is flagged as client-bound.
- **Closed loop, fixed windows.** C virtual users each send their next request as soon as the last one returns. Every run has a warmup at the target concurrency (discarded; it also opens all connections), then a fixed measurement window. Only requests that start inside the window count.
- **Same timing for every protocol.** The request body is pre-encoded once. The timer covers send → full response → response decoded.

**Environment:** Docker Desktop on Apple Silicon (10 vCPUs, arm64 Linux VM), torch 2.15 CPU, 1 sample per request. Latency: 1 user, 10 s. Throughput: 16 users, 30 s. Concurrency: 10 s per level. These are single runs, so expect roughly ±10% between runs. For example, `fastapi_batched` at 16 users did 798 RPS in the throughput bench and 720 RPS in the sweep.

In every run below, client CPU stayed at or below 5%, so the numbers measure the servers.

## Benchmark Results

### Latency (1 user, sequential, 10 s)

| Server | P50 | P95 | P99 | RPS |
|---|---:|---:|---:|---:|
| fastapi | 5.04 ms | 5.22 ms | 7.61 ms | 195.7 |
| grpc | 4.96 ms | 7.94 ms | 9.36 ms | 184.5 |
| fastapi_queue | 2.88 ms | 2.96 ms | 3.06 ms | 342.6 |
| fastapi_msgpack | 6.39 ms | 6.59 ms | 7.57 ms | 155.5 |
| **go** | **2.16 ms** | 5.66 ms | 7.51 ms | **408.3** |
| fastapi_batched | 5.03 ms | 5.15 ms | 5.90 ms | 197.5 |

### Throughput (16 users, 30 s)

| Server | P50 | P95 | P99 | RPS |
|---|---:|---:|---:|---:|
| fastapi | 42.99 ms | 49.00 ms | 59.87 ms | 364.9 |
| grpc | 48.65 ms | 51.09 ms | 59.19 ms | 327.9 |
| fastapi_queue | 42.15 ms | 81.90 ms | 87.14 ms | 344.3 |
| fastapi_msgpack | 100.95 ms | 107.25 ms | 116.42 ms | 157.8 |
| go | 30.39 ms | 41.70 ms | 49.18 ms | 516.6 |
| **fastapi_batched** | **19.92 ms** | **20.40 ms** | **26.30 ms** | **798.4** |

### Concurrency sweep (10 s per level)

**Throughput (RPS)**

| Users | fastapi | grpc | fastapi_queue | fastapi_msgpack | go | fastapi_batched |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 196 | 197 | 335 | 156 | **417** | 190 |
| 4 | 325 | 309 | 340 | 151 | **539** | 232 |
| 8 | 398 | 320 | 346 | 154 | 537 | **573** |
| 16 | 371 | 320 | 350 | 154 | 541 | **720** |
| 32 | 335 | 321 | 350 | 155 | 558 | **1581** |
| 64 | 310 | 322 | 341 | 153 | 530 | **1779** |

**Latency P50 / P99 (ms)**

| Users | fastapi | grpc | fastapi_queue | fastapi_msgpack | go | fastapi_batched |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.1 / 5.5 | 5.1 / 6.0 | 3.0 / 3.3 | 6.4 / 6.9 | **2.0** / 7.3 | 5.0 / 15.2 |
| 4 | 12.3 / 13.9 | 12.5 / 40.7 | 11.6 / 17.8 | 25.4 / 75.8 | **7.1** / 18.6 | 17.1 / 20.7 |
| 8 | 19.8 / 23.8 | 24.9 / 28.8 | 19.9 / 40.5 | 51.5 / 58.6 | 14.8 / 27.3 | **13.9 / 16.6** |
| 16 | 43.0 / 50.4 | 49.8 / 56.9 | 42.2 / 84.4 | 102.8 / 111.5 | 29.3 / 43.6 | **22.1 / 31.3** |
| 32 | 92.6 / 243.9 | 99.4 / 110.3 | 87.4 / 175.1 | 205.9 / 222.9 | 56.9 / 79.6 | **19.8 / 34.7** |
| 64 | 196.0 / 421.3 | 198.3 / 213.5 | 181.8 / 361.6 | 411.9 / 428.8 | 116.5 / 292.3 | **35.2 / 51.6** |

### Forward-pass cost inside the server container

Model only, no HTTP, `cpuset 0-3`, ms per forward pass:

| Where / threads | batch 1 | batch 16 | batch 64 |
|---|---:|---:|---:|
| main thread, 4 threads (default) | 2.56 | 12.21 | 22.11 |
| worker thread, 4 threads | 4.75 | 15.28 | 25.70 |
| worker thread, 2 threads | 6.49 | 25.96 | 43.32 |
| main thread, 1 thread | 8.55 | 44.59 | 77.22 |

A batch of 4 costs about as much as 4 single passes (9.9 ms vs 2.7 ms), and more than a batch of 8 (6.4 ms). The per-sample cost only falls from a batch of about 16 upward.

## Key Takeaways

1. **The old numbers measured the setup, not the servers.** The earlier runs used a single-process httpx client on the host, sharing CPUs with the servers and going through port forwarding. They showed FastAPI at 87 RPS with an 11 ms P50 for 1 user, and a multi-second P99 at 32 users. Isolated, the same server does 196 RPS at 5 ms, with a 244 ms P99 at 32 users.

2. **Protocol barely matters.** FastAPI/JSON and gRPC are identical for 1 user (5.0 ms). gRPC's flat ~320 RPS with linearly growing latency is just `max_workers=4` queueing. FastAPI peaks at about 400 RPS at 8 users, then declines and its tail grows, because its 40-thread pool runs many forward passes at once. msgpack is the slowest at every level, at ~155 RPS.

3. **Which thread runs inference matters.** On this torch/Linux ARM build, a forward pass on a worker thread is ~1.8× slower at batch 1 than on the main thread. That penalty hits fastapi (thread pool), grpc (thread pool) and fastapi_batched (executor). fastapi_queue runs inference on the main event-loop thread, so it gets 2.9 ms for 1 user but stays at 340 RPS. This doesn't explain fastapi_msgpack, which also runs on the main thread but is slower.

4. **Go + ONNX Runtime is best at low concurrency** (2.0 ms P50, 417 RPS for 1 user). It levels off at about 540 RPS because a mutex serializes its single session at batch size 1.

5. **Dynamic batching wins under load.** From 8 users up, `fastapi_batched` leads on both throughput and tail latency. At 64 users it does **1,779 RPS with a 52 ms P99**: 3.4× Go (530 RPS, 292 ms P99) and 5.7× FastAPI (310 RPS, 421 ms P99). Its latency stays nearly flat from 16 to 64 users while every other server's grows linearly. For 1–4 users it's no better than FastAPI, because of the worker-thread penalty and because batches that small don't amortize.

## Known Limitations

- **Single run per configuration,** with no repeats or confidence intervals.
- **Closed-loop load understates tail latency** under overload (coordinated omission). An open-loop, fixed-arrival-rate mode would show it better.
- **Not a real network.** Docker Desktop runs Linux in a VM on macOS, and client and server share one machine and its memory bandwidth. The cpusets map to VM vCPUs, not physical P/E cores.
- **The Go server ignores the cpuset.** ONNX Runtime sizes its thread pool to all 10 VM cores and logs `pthread_setaffinity_np failed`. Setting intra-op threads explicitly in `servers/go/main.go` would fix it.
- **The Go image can't be rebuilt right now.** `servers/go/Dockerfile` downloads `onnxruntime-linux-x64-1.19.2.tgz`, which now returns 404 and is the wrong architecture for arm64 anyway. These results used the previously built image.

## Quick Start

```bash
# Install deps + train model (one-time)
uv sync && make train

# Build and start servers
make build && make up

# Isolated benchmarks (fresh container per server, client in Docker on its own cores)
make bench-all                                   # all benches → results/all.html
make compare                                     # concurrency sweep → results/concurrency.html
make bench-docker BENCH=latency SERVERS="go fastapi_batched"   # subset

# Quick run from the host against running servers (shares CPUs -- not for comparisons)
make up && make bench BENCH=latency SERVER=fastapi && make down
```

## Project Structure

```
model/          Model definition + training script
servers/        Server implementations (one directory per server)
benches/        Benchmark suite (latency, throughput, concurrency) + client Dockerfile
compose/        Docker Compose files per server + bench.yml (client container)
results/        Bench output (git-ignored)
runner.py       CLI benchmark runner
report.py       Results → table/JSON/HTML
config.yaml     Server URLs + benchmark parameters
```

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

## Benchmark Results

The latency, throughput and concurrency tables below were run on Docker Desktop (CPU-only), single-container per server. The dynamic-batching section was measured separately, natively on the host (see there).

### Latency (sequential, 1 user, 200 requests)

| Server | P50 | P95 | P99 | RPS |
|---|---:|---:|---:|---:|
| fastapi | 10.97 ms | 13.94 ms | 17.27 ms | 87.1 |
| **grpc** | **8.89 ms** | **9.23 ms** | **9.29 ms** | **112.5** |
| fastapi_queue | 9.15 ms | 15.30 ms | 27.62 ms | 99.8 |

### Throughput (sustained load, 16 concurrent, 30s)

| Server | P50 | P95 | P99 | RPS |
|---|---:|---:|---:|---:|
| fastapi | 60.12 ms | 484.67 ms | 935.60 ms | 121.8 |
| **grpc** | **74.02 ms** | **90.58 ms** | **102.27 ms** | **206.9** |
| fastapi_queue | 3498.35 ms | 8706.80 ms | 11452.33 ms | 3.8 |

### Concurrency Sweep (latency vs concurrent users)

**FastAPI (REST)**
| Concurrency | P50 | P95 | RPS |
|---:|---:|---:|---:|
| 1 | 68.96 ms | 71.81 ms | 14.5 |
| 4 | 207.90 ms | 307.85 ms | 17.8 |
| 8 | 744.14 ms | 1340.44 ms | 9.0 |
| 16 | 1432.75 ms | 4315.40 ms | 8.0 |
| 32 | 3762.00 ms | 8688.33 ms | 6.9 |

**gRPC**
| Concurrency | P50 | P95 | RPS |
|---:|---:|---:|---:|
| 1 | 8.65 ms | 10.61 ms | 111.2 |
| 4 | 18.00 ms | 18.69 ms | 220.0 |
| 8 | 35.01 ms | 36.53 ms | 225.9 |
| 16 | 71.32 ms | 84.40 ms | 215.7 |
| 32 | 140.61 ms | 143.29 ms | 223.6 |

**FastAPI Queue**
| Concurrency | P50 | P95 | RPS |
|---:|---:|---:|---:|
| 1 | 115.04 ms | 123.47 ms | 8.6 |
| 4 | 481.59 ms | 930.61 ms | 7.1 |
| 8 | 1183.86 ms | 3864.13 ms | 4.4 |
| 16 | 3753.63 ms | 10297.64 ms | 3.5 |
| 32 | 7425.69 ms | 21063.21 ms | 3.2 |

### Dynamic batching (`fastapi_batched` vs `fastapi`)

Measured separately from the Docker runs above: servers run natively on an Apple Silicon Mac (10 cores, 4 performance cores), torch 2.15 CPU, Python 3.14. The numbers are **not comparable** with the Docker tables above. They only compare the two servers with each other.

**Forward pass cost by batch size** (model only, no HTTP):

| Batch | 1 thread | 4 threads | 10 threads |
|---:|---:|---:|---:|
| 1 | 0.58 ms (0.58/sample) | 0.57 ms (0.57/sample) | 0.57 ms (0.57/sample) |
| 4 | 2.05 ms (0.51/sample) | 2.05 ms (0.51/sample) | 2.19 ms (0.55/sample) |
| 16 | 3.57 ms (0.22/sample) | 4.79 ms (0.30/sample) | 6.38 ms (0.40/sample) |
| 64 | 10.10 ms (0.16/sample) | 10.35 ms (0.16/sample) | 14.64 ms (0.23/sample) |

Batching only pays off from roughly 16 samples upward. Extra intra-op threads never help, and all 10 cores (efficiency cores included) are slower. This is why `fastapi_batched` defaults to `TORCH_NUM_THREADS=1` and scales by adding processes (`WORKERS`).

**Multi-process load generator** (4 client processes, 8 s per level, 1 sample per request):

| Concurrency | `fastapi` | `fastapi_batched` (1 worker) | `fastapi_batched` (4 workers) |
|---:|---|---|---|
| 1 | 875 RPS, P50 1.14 / P99 1.29 ms | 860 RPS, P50 1.15 / P99 1.25 ms | 857 RPS, P50 1.16 / P99 1.26 ms |
| 16 | 3109 RPS, P50 4.73 / P99 17.46 ms | 3063 RPS, P50 5.09 / P99 **7.46 ms** | **3567 RPS**, P50 4.74 / P99 8.20 ms |
| 32 | 3176 RPS, P50 8.84 / P99 33.02 ms | **4196 RPS**, P50 7.38 / P99 **11.86 ms** | 3565 RPS, P50 8.03 / P99 28.28 ms |
| 64 | 2002 RPS, P50 13.93 / P99 184.77 ms | 2805 RPS, P50 13.03 / P99 141.38 ms | 1896 RPS, P50 14.65 / P99 194.37 ms |

At 64 users the client processes and the server compete for the same cores, so treat that row as unreliable.

**Repo runner** (`runner.py --bench concurrency`, single client process):

| Concurrency | `fastapi` | `fastapi_batched` (1 worker) | `fastapi_batched` (4 workers) |
|---:|---|---|---|
| 1 | 817 RPS, P99 1.37 ms | 839 RPS, P99 1.25 ms | 837 RPS, P99 1.31 ms |
| 4 | 2106 RPS, P99 3.75 ms | 1076 RPS, P99 5.38 ms | 1902 RPS, P99 5.02 ms |
| 8 | 1287 RPS, P99 10.40 ms | 1263 RPS, P99 18.59 ms | 1445 RPS, P99 14.12 ms |
| 16 | 1234 RPS, P99 29.88 ms | 1093 RPS, P99 34.23 ms | 1134 RPS, P99 36.00 ms |
| 32 | 1036 RPS, P99 87.26 ms | 994 RPS, P99 71.02 ms | 1046 RPS, P99 88.44 ms |

Every configuration levels off at about 1,000–1,300 RPS here. The single-process httpx client is the bottleneck, so this bench can't tell the servers apart at high concurrency.

## Key Takeaways

1. **gRPC dominates across the board** in the Docker runs. It achieves the lowest latency (8.9 ms P50), highest throughput (207 RPS), and scales near-linearly -- maintaining ~220 RPS from 4 to 32 concurrent users with tight tail latencies. Much of this comes from `max_workers=4` capping concurrent forward passes: beyond 4 users, requests queue and latency grows linearly while RPS stays flat.

2. **FastAPI REST is a solid baseline** for low-concurrency use cases. Single-user latency (11 ms P50) is competitive, but latency degrades sharply under load. `predict` is a sync `def`, so it runs in FastAPI's thread pool (up to 40 threads). Many concurrent forward passes, each using torch's own threads, oversubscribe the CPU. RPS actually *drops* as concurrency increases.

3. **The async queue pattern is the slowest.** The single-threaded `asyncio.Queue` worker serializes all inference, creating a bottleneck. At 16 concurrent users it's ~60x slower than gRPC. This pattern only makes sense when you need fire-and-forget semantics or downstream batching.

4. **Tail latencies reveal the real story.** FastAPI's P99 balloons from 17 ms (1 user) to 10+ seconds (32 users). gRPC's P99 stays under 145 ms at 32 concurrent users -- a 70x difference.

5. **Dynamic batching wins under load.** In the local multi-process test, one `fastapi_batched` process handled 32 concurrent users at 4,196 RPS with an 11.9 ms P99. The baseline managed 3,176 RPS with a 33.0 ms P99, so that's 32% more throughput and about a third of the tail latency. At 1 user the two are identical, so batching adds no delay at low load. Batching works because a single inference thread avoids CPU oversubscription and batches grow on their own while a forward pass runs.

6. **Benchmark the load generator too.** A single-process Python client tops out around 1,000–1,300 RPS against servers that can do more than 4,000. For fair results, use a multi-process client, run it inside the Docker network, and pin `cpus:` per container.

## Quick Start

```bash
# Install deps + train model (one-time)
uv sync && make train

# Build and start servers
make build && make up

# Run benchmarks
make bench BENCH=latency SERVER=all    # single bench
make bench-all                          # full matrix
make compare                            # concurrency sweep → HTML report

# Tear down
make down
```

## Project Structure

```
model/          Model definition + training script
servers/        Server implementations (one directory per server)
benches/        Benchmark suite (latency, throughput, concurrency)
compose/        Docker Compose files per server
runner.py       CLI benchmark runner
report.py       Results → table/JSON/HTML
config.yaml     Server URLs + benchmark parameters
```

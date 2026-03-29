# ML Inference Bench

Benchmark and compare three PyTorch serving strategies for a vessel track predictor LSTM model (~3 MB, hidden=256, layers=2).

## Servers

| Server | Protocol | Port | Pattern |
|---|---|---|---|
| `fastapi` | HTTP/REST | 8001 | Sync inference in async handler |
| `grpc` | gRPC/HTTP2 | 8002 | Binary protocol, thread pool (4 workers) |
| `fastapi_queue` | HTTP/REST | 8003 | Submit → poll via `asyncio.Queue` |

**Model:** LSTM encoder → linear decoder
- Input: 30 steps x 5 features (150 floats)
- Output: 15 steps x 2 features (30 floats)

## Benchmark Results

All benchmarks run on Docker Desktop (CPU-only), single-container per server.

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

## Key Takeaways

1. **gRPC dominates across the board.** It achieves the lowest latency (8.9 ms P50), highest throughput (207 RPS), and scales near-linearly -- maintaining ~220 RPS from 4 to 32 concurrent users with tight tail latencies.

2. **FastAPI REST is a solid baseline** for low-concurrency use cases. Single-user latency (11 ms P50) is competitive, but latency degrades sharply under load due to sync inference blocking the event loop. RPS actually *drops* as concurrency increases.

3. **The async queue pattern is the slowest.** The single-threaded `asyncio.Queue` worker serializes all inference, creating a bottleneck. At 16 concurrent users it's ~60x slower than gRPC. This pattern only makes sense when you need fire-and-forget semantics or downstream batching.

4. **Tail latencies reveal the real story.** FastAPI's P99 balloons from 17 ms (1 user) to 10+ seconds (32 users). gRPC's P99 stays under 145 ms at 32 concurrent users -- a 70x difference.

## Quick Start

```bash
# Train model (one-time)
cd model && uv run train.py && cd ..

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
servers/        Three server implementations (fastapi, grpc, fastapi_queue)
benches/        Benchmark suite (latency, throughput, concurrency)
compose/        Docker Compose files per server
runner.py       CLI benchmark runner
report.py       Results → table/JSON/HTML
config.yaml     Server URLs + benchmark parameters
```

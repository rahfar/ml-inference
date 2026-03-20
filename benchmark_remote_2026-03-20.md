# Remote Benchmark Results — 2026-03-20

## Setup

| Component       | Details                                           |
| --------------- | ------------------------------------------------- |
| Server          | VPS x86_64 Linux (4 VCPU, 8GB RAM)                |
| Client          | macOS Darwin 25.3.0 (local laptop, over internet) |
| Python (server) | 3.12.3                                            |
| Python (client) | 3.14                                              |
| Duration        | 20 s                                              |
| Concurrency     | 10 async workers                                  |
| Batch size      | 100 vessels/request                               |
| Model           | LSTM encoder (hidden=256, layers=2), ~3 MB        |

### Server configurations

**FastAPI** — 1 uvicorn worker, 1-thread pool executor, orjson responses, async endpoints with `run_in_executor`, raw `request.json()` (no Pydantic validation on hot path). PyTorch uses all CPU cores internally for batch inference, so a single thread is sufficient.

**gRPC** — ThreadPoolExecutor with 4 workers, binary protobuf serialization.

## Latency & throughput

| Server   |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | requests | errors |
| -------- | -------: | --------: | ------: | ------: | ------: | ------: | -------: | -----: |
| FastAPI  |     34.7 |     3 467 |     277 |     388 |     618 |   1 022 |      701 |      0 |
| **gRPC** | **41.5** | **4 153** | **238** | **313** | **380** | **510** |      841 |      0 |

## Data throughput

| Server   | req payload | resp payload | round-trip | total transferred | throughput |
| -------- | ----------: | -----------: | ---------: | ----------------: | ---------: |
| FastAPI  |    250.1 KB |      28.9 KB |   279.0 KB |            190 MB |   9.5 MB/s |
| **gRPC** |     79.4 KB |      17.3 KB |    96.7 KB |             79 MB |   4.0 MB/s |

Protobuf compression ratios vs JSON: **3.2×** smaller requests, **1.7×** smaller responses.

## Key takeaways

- gRPC wins by **~20%** on throughput (41.5 vs 34.7 req/s) and has significantly tighter tail latency (p99: 380 vs 618 ms).
- FastAPI pushes **2.4× more bytes** over the wire yet completes **17% fewer requests** — JSON serialization overhead compounds under concurrency.
- Network round-trip dominates: local benchmarks show ~70 ms p50 for both servers; over the network p50 jumps to 240–277 ms.
- Single-thread pool for FastAPI performs identically to multi-thread — PyTorch saturates all CPU cores within a single batch inference call.

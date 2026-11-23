# CLAUDE.md

## Project Overview

ML Inference Bench — benchmark and compare three PyTorch serving strategies for a vessel track predictor LSTM model.

**Servers:**
| Server | Protocol | Host Port | Pattern |
|---|---|---|---|
| `fastapi` | HTTP/REST | 8001 | Sync inference in async handler |
| `grpc` | gRPC/HTTP2 | 8002 | Binary protocol, thread pool |
| `fastapi_queue` | HTTP/REST | 8003 | Submit → poll via asyncio.Queue |
| `fastapi_msgpack` | HTTP/MessagePack | 8004 | Inline inference, msgpack body |
| `go` | HTTP/REST | 8005 | Go + ONNX Runtime |
| `fastapi_batched` | HTTP/REST | 8006 | In-process dynamic batching (torch); env: WORKERS, TORCH_NUM_THREADS, MAX_BATCH, MAX_QUEUE, REQUEST_TIMEOUT_S |

**Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)
- Input: 30 steps × 5 features (flat 150 floats)
- Output: 15 steps × 2 features (flat 30 floats)

## Repo Structure

```
model/
  predictor.py          # VesselTrackPredictor + dimension constants
  train.py              # Synthetic data generation + training
  weights/model.pt      # Saved weights (git-ignored)
servers/
  fastapi/app.py        # Baseline REST server
  grpc/server.py        # gRPC server + proto/stubs
  fastapi_queue/app.py  # Queue-based server (submit/poll)
  fastapi_msgpack/app.py # MessagePack serialization
  fastapi_batched/app.py # Dynamic batching (asyncio.Queue + Future, 1 inference thread)
  go/main.go            # Go + ONNX Runtime
benches/
  base.py               # Protocol clients, multi-process load runner, BenchResult
  Dockerfile            # Load-generator image (no torch)
  latency_bench.py      # Single-user P50/P95/P99
  throughput_bench.py   # Max RPS under sustained load
  concurrency_bench.py  # Latency vs concurrent users curve
compose/
  compose.<server>.yml  # One file per server (cpuset SERVER_CPUS, default 0-3)
  bench.yml             # Bench client container (cpuset CLIENT_CPUS, default 4-9)
runner.py               # CLI bench runner (--target host|docker)
report.py               # Results → table/JSON/HTML; CLI merges JSON files
results/                # Bench output (git-ignored)
config.yaml             # Server URLs, bench defaults
Makefile                # Developer shortcuts
```

## Commands

**Package manager:** `uv`

```bash
# Install deps (runtime + dev group: grpcio-tools, onnxscript, ruff, ty)
uv sync

# Train model (required before running servers; also exports model.onnx for Go)
make train

# Regenerate gRPC stubs
make proto

# Docker workflow
make build          # build all server images
make up             # start all servers (waits for healthy)
make down           # tear down

# Real numbers: isolated Docker run (fresh container per server, client on the
# compose network with its own cpuset) -> results/<bench>-<server>.json + results/<bench>.html
make bench-docker BENCH=latency                     # SERVERS="fastapi go" to subset
make compare        # = bench-docker BENCH=concurrency
make bench-all      # = bench-docker BENCH=all

# Quick host run against already-running servers (client shares CPUs with servers)
make bench BENCH=latency SERVER=fastapi
uv run python runner.py --bench concurrency --server fastapi --concurrency 1 16 --duration 5
```

## Architecture

### Bench suite
- `latency_bench` — 1 user, sequential, fixed duration
- `throughput_bench` — fixed concurrency, fixed duration
- `concurrency_bench` — sweeps concurrency levels, each an independent warmup + measurement window

All benches call `BaseBench.load()` → `run_load()`: closed-loop users spread over up to `client.processes` spawned processes (aiohttp/uvloop or grpc.aio), warmup window discarded, only requests starting in the measurement window count. Request bodies are pre-encoded per protocol; timing covers send → response decoded. Each result reports client CPU of the busiest client process — ≥80% means the run is client-bound.

### Config
All server URLs (`url` for host, `docker_url` for the compose network), client processes and bench parameters live in `config.yaml`. The runner reads from there — no hardcoded URLs in bench code. Queue-pattern servers are marked with `pattern: queue`.

No test framework is configured. Lint/format with `make style` (ruff).

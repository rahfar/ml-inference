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
| `fastapi_batched` | HTTP/REST | 8006 | In-process dynamic batching (torch); env: WORKERS, TORCH_NUM_THREADS, MAX_BATCH, MAX_QUEUE, REQUEST_TIMEOUT_S |

**Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)
- Input: 30 steps × 5 features (flat 150 floats)
- Output: 15 steps × 2 features (flat 30 floats)

## Repo Structure

```
model/
  predictor.py          # VesselTrackPredictor, dimension constants, WEIGHTS_PATH
  train.py              # Synthetic data generation + training
  weights/model.pt      # Saved weights (git-ignored)
servers/
  fastapi/app.py        # Baseline REST server
  grpc/server.py        # gRPC server + proto/stubs
  fastapi_queue/app.py  # Queue-based server (submit/poll)
  fastapi_msgpack/app.py # MessagePack serialization
  fastapi_batched/app.py # Dynamic batching (asyncio.Queue + Future, 1 inference thread)
benches/
  base.py               # Protocol clients, multi-process load runner, BenchResult
  server_proc.py        # Start/stop a server subprocess per bench run (--spawn)
  latency_bench.py      # Single-user P50/P95/P99
  throughput_bench.py   # Max RPS under sustained load
  concurrency_bench.py  # Latency vs concurrent users curve
runner.py               # CLI bench runner
report.py               # Results → table/JSON/HTML; CLI merges JSON files
config.yaml             # Server URLs, start commands, bench defaults
results/                # Bench output + server logs (git-ignored)
Makefile                # Developer shortcuts
```

## Commands

**Package manager:** `uv`. Everything runs locally (no Docker).

```bash
# Install deps (runtime + dev group: grpcio-tools, ruff, ty)
uv sync

# Train model (required before running servers)
make train

# Regenerate gRPC stubs
make proto

# Benchmarks -- each run starts a fresh server process (runner.py --spawn)
make bench BENCH=latency SERVER=fastapi     # -> results/latency.html
make compare                                # concurrency sweep -> results/concurrency.html
make bench-all                              # full matrix -> results/all.html

# Run a server by hand, bench it without --spawn
make serve-fastapi_batched
uv run python runner.py --bench concurrency --server fastapi_batched --concurrency 1 16 --duration 5
```

## Architecture

### Bench suite
- `latency_bench` — 1 user, sequential, fixed duration
- `throughput_bench` — fixed concurrency, fixed duration
- `concurrency_bench` — sweeps concurrency levels, each an independent warmup + measurement window

All benches call `BaseBench.load()` → `run_load()`: closed-loop users spread over up to `client.processes` spawned processes (aiohttp/uvloop or grpc.aio), warmup window discarded, only requests starting in the measurement window count. Request bodies are pre-encoded per protocol; timing covers send → response decoded. Each result reports client CPU of the busiest client process — ≥80% means the run is client-bound. Client and servers share the host's CPUs (no pinning on macOS).

### Config
All server URLs, start commands (`cmd`, used by `--spawn`), client processes and bench parameters live in `config.yaml`. The runner reads from there — no hardcoded URLs in bench code. Queue-pattern servers are marked with `pattern: queue`.

No test framework is configured. Lint/format with `make style` (ruff).

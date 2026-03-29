# CLAUDE.md

## Project Overview

ML Inference Bench — benchmark and compare three PyTorch serving strategies for a vessel track predictor LSTM model.

**Three servers:**
| Server | Protocol | Host Port | Pattern |
|---|---|---|---|
| `fastapi` | HTTP/REST | 8001 | Sync inference in async handler |
| `grpc` | gRPC/HTTP2 | 8002 | Binary protocol, thread pool |
| `fastapi_queue` | HTTP/REST | 8003 | Submit → poll via asyncio.Queue |

**Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)
- Input: 30 steps × 5 features (flat 150 floats)
- Output: 15 steps × 2 features (flat 30 floats)

## Repo Structure

```
model/
  model.py              # VesselTrackPredictor + dimension constants
  train.py              # Synthetic data generation + training
  weights/model.pt      # Saved weights (git-ignored)
servers/
  fastapi/app.py        # Baseline REST server
  grpc/server.py        # gRPC server + proto/stubs
  fastapi_queue/app.py  # Queue-based server (submit/poll)
benches/
  base.py               # BaseBench ABC + BenchResult dataclass
  latency_bench.py      # Single-user P50/P95/P99
  throughput_bench.py   # Max RPS under sustained load
  concurrency_bench.py  # Latency vs concurrent users curve
compose/
  docker-compose.yml    # All 3 servers
runner.py               # CLI bench runner (runs on host)
report.py               # Results → table/JSON/HTML
config.yaml             # Server URLs, bench defaults
Makefile                # Developer shortcuts
```

## Commands

**Package manager:** `uv`

```bash
# Train model (required before running servers)
cd model && uv run train.py

# Regenerate gRPC stubs
make proto

# Docker workflow
make build          # build all 3 server images
make up             # start all servers (waits for healthy)
make down           # tear down

# Run benchmarks (servers must be running)
make bench BENCH=latency SERVER=fastapi
make bench BENCH=concurrency SERVER=all
make compare        # concurrency sweep → HTML report
make bench-all      # full matrix → HTML report

# Or directly:
python runner.py --bench latency --server fastapi
python runner.py --bench all --server all --output html
```

## Architecture

### Bench suite
- `latency_bench` — N sequential requests, 1 user, reports P50/P95/P99
- `throughput_bench` — sustained concurrent load for duration, reports max stable RPS
- `concurrency_bench` — sweeps concurrency levels, reports latency curve

All benches inherit from `BaseBench` which handles HTTP, gRPC, and queue (submit+poll) request patterns.

### Config
All server URLs and bench parameters live in `config.yaml`. The runner reads from there — no hardcoded URLs in bench code.

No test framework or linter is configured.

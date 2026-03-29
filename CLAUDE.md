# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML inference benchmark serving a **PyTorch LSTM** vessel track predictor via three server implementations:
- **FastAPI direct** — thread-pool inference, no queue
- **FastAPI + RQ queue** — Redis-backed job queue, separate worker process
- **gRPC** — binary protocol, thread-pool

- **Task:** predict 15-min future vessel trajectory from 5-min AIS history
- **Input:** 30 track points × 5 features (lat, lon, speed, course\_sin, course\_cos) at 10-sec intervals
- **Output:** 15 waypoints × 2 features (lat, lon) at 1-min intervals
- **Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)

## Commands

**Package manager:** `uv`

```bash
# Train model (required before running servers)
uv run train_pytorch.py

# Regenerate gRPC stubs (only needed after editing services/grpc/inference.proto)
uv run python -m grpc_tools.protoc -I services/grpc --python_out=services/grpc --grpc_python_out=services/grpc services/grpc/inference.proto

# Start a server manually (all run from project root)
uv run python services/fastapi_direct/server.py      # --port NNNN
uv run python services/grpc/server.py                # --port NNNN
uv run python services/fastapi_queue/server.py       # --port NNNN --redis-url redis://localhost:6379
uv run python services/fastapi_queue/rq_worker.py    # start in a separate terminal

# Run the full benchmark (starts/stops servers automatically)
uv run benchmarks/run.py --server all

# Quick sanity check
uv run benchmarks/run.py --server grpc --config benchmarks/config/quick.yaml

# Benchmark a single server
uv run benchmarks/run.py --server fastapi_direct --duration 30 --concurrency 20
uv run benchmarks/run.py --server fastapi_queue  --duration 30 --concurrency 20
uv run benchmarks/run.py --server grpc           --duration 30 --concurrency 20

# Connect to already-running server (no spawn)
uv run benchmarks/run.py --server fastapi_direct --no-spawn --port 8001
```

`benchmarks/run.py` flags: `--server [fastapi_direct|fastapi_queue|grpc|all]`, `--config` (YAML file, default `benchmarks/config/default.yaml`), `--duration` / `--concurrency` / `--batch-size` (override config), `--no-spawn`, `--no-save`, `--json`.

Config files live in `benchmarks/config/`. Results are auto-saved to `benchmarks/results/` as timestamped JSON.

**fastapi_queue prerequisite:** Redis must be running before the benchmark starts. `benchmarks/run.py` spawns the worker automatically.

No test framework or linter is configured.

## Architecture

### Model layer (`model_def.py`)
- Defines `HISTORY_STEPS=30`, `HISTORY_FEATURES=5`, `FUTURE_STEPS=15`, `FUTURE_FEATURES=2` — imported by training, servers, and load_test
- `VesselTrackPredictor`: LSTM encoder (input 5 → hidden 256 × 2 layers) + Linear decoder → reshape to (batch, 15, 2)

### Training (`train_pytorch.py`)
- Generates 20K synthetic vessel tracks using dead-reckoning random walk
- Trains 20 epochs with Adam, saves to `models/pytorch_model.pt`

### Service layer (`services/`)

```
services/
  fastapi_direct/server.py     FastAPI + Uvicorn; inference offloaded to ThreadPoolExecutor
  fastapi_queue/
    server.py                  FastAPI + Uvicorn; enqueues jobs to Redis, async-polls for result
    inference_task.py          RQ job function; loads model lazily, runs batch inference
    rq_worker.py               Worker process launcher
  grpc/server.py               gRPC threadpool (4 workers)
```

All services expose identical HTTP semantics (fastapi_*):
- `GET  /health` → `{"status": "ok"}`
- `POST /predict` → `{"history": [{lat,lon,speed,course_sin,course_cos}×30]}` → `{"prediction": [[lat,lon]×15]}`
- `POST /predict_batch` → `{"vessels": [<history>×N]}` → `{"predictions": [[[lat,lon]×15]×N]}`

gRPC: `Inference.Predict` / `Inference.PredictBatch`; proto in `inference.proto`

### Benchmark layer (`benchmarks/`)

```
benchmarks/
  run.py                CLI entrypoint
  config/
    default.yaml        batch_size, concurrency, duration, timeouts, etc.
    quick.yaml          fast sanity-check values
  harness/
    lifecycle.py        server spawn/teardown, health polling
    monitor.py          CPU/RSS sampling thread (per-process, by label)
    payloads.py         build HTTP and gRPC test payloads
  runners/
    http_runner.py      httpx async load loop
    grpc_runner.py      grpc.aio async load loop
  metrics/
    stats.py            percentiles (linear interpolation), aggregation
  results/              auto-saved JSON per run (timestamp + git SHA in filename)
```

Flow:
1. Load YAML config; CLI flags override individual keys
2. Spawn server subprocess(es); for `fastapi_queue` also spawns worker
3. Poll `/health` or gRPC `Health` until ready; optionally run warmup
4. Background monitor samples RSS+CPU for each process separately
5. Drive async load via runner; collect raw latency list
6. Compute p50/p95/p99 (interpolated), req/s, vessels/s; print summary
7. Save full result JSON to `benchmarks/results/`

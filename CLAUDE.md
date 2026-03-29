# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML inference benchmark serving a **PyTorch LSTM** vessel track predictor via two server implementations:
- **HTTP** — FastAPI + Uvicorn, thread-pool inference
- **gRPC** — binary protocol, thread-pool

- **Task:** predict 15-min future vessel trajectory from 5-min AIS history
- **Input:** 30 track points × 5 features (lat, lon, speed, course\_sin, course\_cos) at 10-sec intervals
- **Output:** 15 waypoints × 2 features (lat, lon) at 1-min intervals
- **Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)

## Commands

**Package manager:** `uv`

```bash
# Train model (required before running servers)
uv run train.py

# Regenerate gRPC stubs (only needed after editing inference.proto)
uv run python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. inference.proto

# Start a server manually
uv run server_http.py      # --host 0.0.0.0 --port 8000
uv run server_grpc.py      # --host 0.0.0.0 --port 8000

# Run the full benchmark (starts/stops servers automatically)
uv run benchmarks/run.py --server all

# Quick sanity check
uv run benchmarks/run.py --server grpc --config benchmarks/config/quick.yaml

# Benchmark a single server
uv run benchmarks/run.py --server http --duration 30 --concurrency 20
uv run benchmarks/run.py --server grpc --duration 30 --concurrency 20

# Connect to already-running server
uv run benchmarks/run.py --server http --no-spawn --port 8001
```

`benchmarks/run.py` flags: `--server [http|grpc|all]`, `--config` (YAML, default `benchmarks/config/default.yaml`), `--duration` / `--concurrency` / `--batch-size` (override config), `--no-spawn`, `--no-save`, `--json`.

No test framework or linter is configured.

## Architecture

### Flat structure
```
model_def.py          # VesselTrackPredictor + sequence dimension constants
train.py              # synthetic data generation + LSTM training
inference.py          # shared inference engine: get_model(), predict(), predict_batch()
server_http.py        # FastAPI server — uses inference.py
server_grpc.py        # gRPC server  — uses inference.py
inference.proto       # Protobuf service definition
inference_pb2.py      # generated stubs
inference_pb2_grpc.py # generated stubs
```

### Model layer (`model_def.py`)
- Defines `HISTORY_STEPS=30`, `HISTORY_FEATURES=5`, `FUTURE_STEPS=15`, `FUTURE_FEATURES=2`
- `VesselTrackPredictor`: LSTM encoder (input 5 → hidden 256 × 2 layers) + Linear decoder → reshape to (batch, 15, 2)

### Training (`train.py`)
- Generates 20K synthetic vessel tracks using dead-reckoning random walk
- Trains 20 epochs with Adam, saves to `models/pytorch_model.pt`

### Inference engine (`inference.py`)
- Singleton model loading (`get_model()`) with CUDA auto-detection
- `predict(history: np.ndarray)` — single vessel, (30,5) → (15,2)
- `predict_batch(histories: np.ndarray)` — batch, (N,30,5) → (N,15,2)
- Shared by both servers; no duplication

### Servers
Both servers load the model via `inference.get_model()` at startup and delegate all computation to `inference.predict()` / `inference.predict_batch()`.

HTTP endpoints:
- `GET  /health` → `{"status": "ok"}`
- `POST /predict` → `{"history": [{lat,lon,speed,course_sin,course_cos}×30]}` → `{"prediction": [[lat,lon]×15]}`
- `POST /predict_batch` → `{"vessels": [<history>×N]}` → `{"predictions": [[[lat,lon]×15]×N]}`

gRPC: `Inference.Predict` / `Inference.PredictBatch` / `Inference.Health`

### Benchmark layer (`benchmarks/`)
```
run.py                CLI entrypoint
config/
  default.yaml        batch_size, concurrency, duration, timeouts
  quick.yaml          fast sanity-check values
harness/
  lifecycle.py        server spawn/teardown, health polling
  monitor.py          CPU/RSS sampling thread (per-process)
  payloads.py         build HTTP and gRPC test payloads
runners/
  http_runner.py      httpx async load loop
  grpc_runner.py      grpc.aio async load loop
metrics/
  stats.py            percentiles (linear interpolation), aggregation
results/              auto-saved JSON per run
```

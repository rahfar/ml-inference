# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML inference benchmark serving a **PyTorch LSTM** vessel track predictor via FastAPI (ASGI/Uvicorn), Flask+Waitress (WSGI), and gRPC (binary/threadpool).

- **Task:** predict 15-min future vessel trajectory from 5-min AIS history
- **Input:** 30 track points × 5 features (lat, lon, speed, course\_sin, course\_cos) at 10-sec intervals
- **Output:** 15 waypoints × 2 features (lat, lon) at 1-min intervals
- **Model:** LSTM encoder (hidden=256, layers=2) → linear decoder (~3 MB)

## Commands

**Package manager:** `uv`

```bash
# Train model (required before running servers)
uv run train_pytorch.py

# Regenerate gRPC stubs (only needed after editing inference.proto)
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

# Start a server manually
uv run server_fastapi.py          # or --port NNNN
uv run server_flask.py
uv run server_grpc.py

# Run the full benchmark (starts/stops servers automatically)
uv run load_test.py --server all --duration 20 --concurrency 10

# Benchmark a single server
uv run load_test.py --server grpc --duration 30 --concurrency 20
```

`load_test.py` flags: `--server [fastapi|flask|grpc|all]`, `--duration` (seconds), `--concurrency` (async workers), `--port` (default 8000).

No test framework or linter is configured.

## Architecture

### Model layer (`model_def.py`)
- Defines `HISTORY_STEPS=30`, `HISTORY_FEATURES=5`, `FUTURE_STEPS=15`, `FUTURE_FEATURES=2` — imported by training, servers, and load_test
- `VesselTrackPredictor`: LSTM encoder (input 5 → hidden 256 × 2 layers) + Linear decoder → reshape to (batch, 15, 2)

### Training (`train_pytorch.py`)
- Generates 20K synthetic vessel tracks using dead-reckoning random walk
- Trains 20 epochs with Adam, saves to `models/pytorch_model.pt`

### Server layer
All servers load the model at startup and expose predict + health:
- `server_fastapi.py` — FastAPI + Pydantic + Uvicorn (async); `GET /health`, `POST /predict`
- `server_flask.py` — Flask + Waitress (4 threads, WSGI); same HTTP endpoints
- `server_grpc.py` — gRPC threadpool (4 workers); `Inference.Predict` + `Inference.Health` RPCs; proto in `inference.proto`, stubs in `inference_pb2.py` / `inference_pb2_grpc.py`

HTTP request: `{"history": [{"lat", "lon", "speed", "course_sin", "course_cos"}, ×30]}`
HTTP response: `{"prediction": [[lat, lon], ×15]}`

### Benchmark orchestrator (`load_test.py`)
1. Spawns server subprocess; polls HTTP `/health` or gRPC `Health` RPC until ready
2. Drives async load — HTTP via `httpx`, gRPC via `grpc.aio` — at configured concurrency
3. Background thread samples RSS and CPU via `psutil` every 0.5 s
4. Computes p50/p95/p99/avg/max latency, prints per-run stats and final summary table

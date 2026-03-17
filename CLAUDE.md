# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML inference benchmark comparing CatBoost vs PyTorch models served via FastAPI (ASGI/Uvicorn), Flask+Waitress (WSGI), and gRPC (binary/threadpool). Input: 3 floats → Output: 1 float (regression). Goal is to measure RPS, latency percentiles, memory, and CPU across all 6 server×model combinations.

## Commands

**Package manager:** `uv`

```bash
# Train models (required before running servers)
uv run train_catboost.py
uv run train_pytorch.py

# Regenerate gRPC stubs (only needed after editing inference.proto)
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto

# Start a server manually
uv run server_fastapi.py --model catboost   # or pytorch
uv run server_flask.py --model catboost     # or pytorch
uv run server_grpc.py --model catboost      # or pytorch

# Run the full benchmark (starts/stops servers automatically)
uv run load_test.py --server all --model all --duration 20 --concurrency 10

# Benchmark a specific combination
uv run load_test.py --server grpc --model pytorch --duration 30 --concurrency 20
```

`load_test.py` flags: `--server [fastapi|flask|grpc|all]`, `--model [catboost|pytorch|all]`, `--duration` (seconds), `--concurrency` (async workers), `--port` (default 8000).

No test framework or linter is configured.

## Architecture

### Model layer
- `model_def.py` — PyTorch MLP definition (3 → 512 → 512 → 1), shared by training and serving
- `train_catboost.py` / `train_pytorch.py` — generate synthetic data (20K samples, seeded), train, save to `models/`

### Server layer
All servers load a model at startup and expose a predict + health interface:
- `server_fastapi.py` — FastAPI + Pydantic + Uvicorn (async); `GET /health`, `POST /predict`
- `server_flask.py` — Flask + Waitress (4 threads, WSGI); same HTTP endpoints
- `server_grpc.py` — gRPC threadpool server (4 workers); `Inference.Predict` + `Inference.Health` RPCs defined in `inference.proto`; stubs in `inference_pb2.py` / `inference_pb2_grpc.py`

### Benchmark orchestrator (`load_test.py`)
1. Spawns a server subprocess; polls HTTP `/health` (HTTP servers) or calls `Inference.Health` via gRPC (gRPC server) until ready
2. Drives async load — HTTP via `httpx`, gRPC via `grpc.aio` — at configured concurrency
3. Background thread samples RSS and CPU via `psutil` every 0.5s
4. Collects latency samples, computes p50/p95/p99/avg/max
5. Prints a formatted results table, then kills the server
6. Iterates over all requested server×model combinations

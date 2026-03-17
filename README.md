# ML Inference Benchmark

Compares inference throughput of a **PyTorch LSTM** vessel track predictor served via **FastAPI** (ASGI), **Flask+Waitress** (WSGI), and **gRPC** (binary, threadpool).

**Task:** predict 15-minute future vessel trajectory from 5-minute AIS history.

| | |
|---|---|
| Input  | 30 track points × 5 features (lat, lon, speed, course\_sin, course\_cos) — 10-sec intervals |
| Output | 15 waypoints × 2 features (lat, lon) — 1-min intervals |
| Model  | LSTM encoder (hidden=256, layers=2) → linear decoder, ~3 MB on disk |

Load test: 20 s duration, 10 concurrent async workers.

## Project structure

```
model_def.py          # VesselTrackPredictor + sequence dimension constants
train_pytorch.py      # synthetic track data generation + LSTM training
server_fastapi.py     # FastAPI (ASGI/uvicorn) server
server_flask.py       # Flask + Waitress (WSGI, 4 threads) server
server_grpc.py        # gRPC server (ThreadPoolExecutor, 4 workers)
inference.proto       # Protobuf service definition
inference_pb2.py      # generated stubs (grpc_tools.protoc)
inference_pb2_grpc.py
load_test.py          # orchestrator: starts server, drives load, prints stats
```

## Reproduce

**1. Install dependencies**

```bash
uv add torch numpy fastapi "uvicorn[standard]" flask waitress psutil httpx grpcio grpcio-tools
```

**2. Train model**

```bash
uv run train_pytorch.py
```

**3. Regenerate gRPC stubs** *(only needed after editing `inference.proto`)*

```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
```

**4. Run a server manually (optional)**

```bash
uv run server_fastapi.py          # http://localhost:8000
uv run server_flask.py            # http://localhost:8000
uv run server_grpc.py             # grpc://localhost:8000
```

**5. Run the full benchmark**

```bash
uv run load_test.py --server all --duration 20 --concurrency 10
```

Or target one server:

```bash
uv run load_test.py --server grpc --duration 30 --concurrency 20
```

### `load_test.py` flags

| Flag            | Default | Description                          |
| --------------- | ------- | ------------------------------------ |
| `--server`      | `all`   | `fastapi`, `flask`, `grpc`, or `all` |
| `--duration`    | `20`    | Test duration in seconds             |
| `--concurrency` | `10`    | Concurrent async workers             |
| `--port`        | `8000`  | Port to bind the server on           |

## Results

Environment: macOS Darwin 25.3.0, Python 3.14, 20 s / 10 workers.

| Server       |       RPS |   p50 ms |   p95 ms |    Mem avg | CPU avg |
| ------------ | --------: | -------: | -------: | ---------: | ------: |
| FastAPI      |     2 006 |     4.91 |     6.20 |     241 MB |    236% |
| Flask        |     2 505 |     3.85 |     4.45 |     227 MB |    242% |
| gRPC         | **4 048** | **2.43** | **2.82** | **223 MB** |    393% |

### Observations

**gRPC vs HTTP**

- gRPC is **~2× faster** (4 048 vs 2 505 RPS) and delivers tighter latency (p95 2.8 ms vs 4.5 ms).
- Binary framing (protobuf) eliminates HTTP text parsing. Persistent connections remove per-request TCP/TLS setup.
- The larger request payload (30 structured points vs a few scalars) amplifies the serialization saving: protobuf encoding is roughly 5–10× more compact and faster than JSON for repeated numeric messages.

**Flask beats FastAPI with a heavy model**

- With a small model (previous MLP), FastAPI (ASGI/async) was ~7% faster.
- With the LSTM, Flask+Waitress (4 OS threads) overtakes FastAPI by ~25%.
- Reason: PyTorch LSTM inference releases the GIL, so Waitress's 4 true threads can run 4 inferences in parallel. FastAPI's single-process event loop cannot overlap CPU-bound compute as effectively, even though it is non-blocking.
- Memory is similar across all three servers (~220–240 MB); the model dominates the footprint.

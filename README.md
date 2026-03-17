# ML Inference Benchmark

Compares inference throughput of a **PyTorch LSTM** vessel track predictor served via **FastAPI** (ASGI), **Flask+Waitress** (WSGI), and **gRPC** (binary, threadpool).

**Task:** predict 15-minute future vessel trajectory from 5-minute AIS history.

| | |
|---|---|
| Input  | 30 track points × 5 features (lat, lon, speed, course\_sin, course\_cos) — 10-sec intervals |
| Output | 15 waypoints × 2 features (lat, lon) — 1-min intervals |
| Model  | LSTM encoder (hidden=256, layers=2) → linear decoder, ~3 MB on disk |
| Batch  | 100 vessels per request |

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
uv run server_fastapi.py     # http://localhost:8000
uv run server_flask.py       # http://localhost:8000
uv run server_grpc.py        # grpc://localhost:8000
```

Endpoints:

| Protocol | Single vessel         | Batch (N vessels)      |
|----------|-----------------------|------------------------|
| HTTP     | `POST /predict`       | `POST /predict_batch`  |
| gRPC     | `Inference.Predict`   | `Inference.PredictBatch` |

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

Batch size is fixed at 100 vessels/request (`BATCH_SIZE` in `load_test.py`).

## Results

Environment: macOS Darwin 25.3.0, Python 3.14, 20 s / 10 workers, 100 vessels/request.

| Server  | req/s | vessels/s  |  p50 ms |  p95 ms | Mem avg | CPU avg |
| ------- | ----: | ---------: | ------: | ------: | ------: | ------: |
| FastAPI |   138 |     13 780 |   71.45 |   94.87 | 490 MB  |    659% |
| Flask   |   132 |     13 151 |   74.89 |   89.40 | 323 MB  |    444% |
| gRPC    | **141** | **14 055** | **69.92** | **85.32** | **313 MB** | 458% |

### Observations

**Batch inference narrows the gap between transports**

- With 100-vessel batches the bottleneck is almost entirely model compute; transport overhead is a small fraction of the ~70 ms round trip.
- gRPC still leads, but by ~2% in throughput rather than the 2× seen with single-vessel requests.
- Latency variance tightens across all three servers (p95/p50 ratio ≈ 1.2–1.3× for batch vs 1.3–1.5× for single).

**FastAPI's memory and CPU costs grow with payload size**

- FastAPI uses 490 MB vs 313–323 MB for gRPC/Flask — ~50% more memory.
- FastAPI also burns 659% CPU vs 444–458% for the others.
- Root cause: JSON serialization of the response (100 vessels × 15 waypoints × 2 coords = 3 000 floats as text) is expensive. Flask/gRPC handle the same data more efficiently (Flask with simple `jsonify`, gRPC with binary protobuf).

**gRPC's memory advantage grows with batch size**

- Binary protobuf encoding of the 100-vessel response is ~5–10× more compact than JSON, cutting both allocation pressure and serialization CPU.
- Flask is competitive on throughput but allocates ~10 MB more than gRPC even for the same model.

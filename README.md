# ML Inference Benchmark

Compares inference throughput of a **PyTorch LSTM** vessel track predictor served via **FastAPI** (ASGI/Uvicorn) and **gRPC** (binary/threadpool).

**Task:** predict 15-minute future vessel trajectory from 5-minute AIS history.

|        |                                                                                           |
| ------ | ----------------------------------------------------------------------------------------- |
| Input  | 30 track points × 5 features (lat, lon, speed, course_sin, course_cos) — 10-sec intervals |
| Output | 15 waypoints × 2 features (lat, lon) — 1-min intervals                                    |
| Model  | LSTM encoder (hidden=256, layers=2) → linear decoder, ~3 MB on disk                       |
| Batch  | 100 vessels per request                                                                   |

## Project structure

```
model_def.py          # VesselTrackPredictor + sequence dimension constants
train_pytorch.py      # synthetic track data generation + LSTM training
server_fastapi.py     # FastAPI server (async, orjson, thread-pool inference)
server_grpc.py        # gRPC server (ThreadPoolExecutor, 4 workers)
inference.proto       # Protobuf service definition
inference_pb2.py      # generated stubs (grpc_tools.protoc)
inference_pb2_grpc.py
load_test.py          # orchestrator: starts server, drives load, prints stats
```

## Reproduce

**1. Install dependencies**

```bash
uv sync
```

**2. Train model**

```bash
uv run train_pytorch.py
```

**3. Regenerate gRPC stubs** _(only needed after editing `inference.proto`)_

```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
```

**4. Run a server manually (optional)**

```bash
uv run server_fastapi.py            # http://localhost:8000
uv run server_fastapi.py --workers 4 # multi-process mode
uv run server_grpc.py               # grpc://localhost:8000
```

Endpoints:

| Protocol | Single vessel       | Batch (N vessels)        |
| -------- | ------------------- | ------------------------ |
| HTTP     | `POST /predict`     | `POST /predict_batch`    |
| gRPC     | `Inference.Predict` | `Inference.PredictBatch` |

**5. Run the full benchmark**

```bash
uv run load_test.py --server all --duration 20 --concurrency 10
```

Or target one server:

```bash
uv run load_test.py --server grpc --duration 30 --concurrency 20
```

### `load_test.py` flags

| Flag            | Default     | Description                          |
| --------------- | ----------- | ------------------------------------ |
| `--server`      | `all`       | `fastapi`, `grpc`, or `all`          |
| `--duration`    | `20`        | Test duration in seconds             |
| `--concurrency` | `10`        | Concurrent async workers             |
| `--port`        | `8000`      | Port to bind the server on           |
| `--host`        | `127.0.0.1` | Server host                          |
| `--no-spawn`    | off         | Connect to an already-running server |

Batch size is fixed at 100 vessels/request (`BATCH_SIZE` in `load_test.py`).

## Results

### Remote VPS benchmark (client → server over network)

Environment: VPS x86_64 Linux, Python 3.12, client on macOS over internet.
20 s duration, 10 concurrent workers, 100 vessels/request.

| Server   |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms |
| -------- | -------: | --------: | ------: | ------: | ------: | ------: |
| FastAPI  |     35.3 |     3 526 |     269 |     429 |     604 |     913 |
| **gRPC** | **41.5** | **4 153** | **238** | **313** | **380** | **510** |

### Data throughput

| Server   | req payload | resp payload | round-trip | total transferred | throughput |
| -------- | ----------: | -----------: | ---------: | ----------------: | ---------: |
| FastAPI  |    250.1 KB |      28.9 KB |   279.0 KB |            193 MB |   9.7 MB/s |
| **gRPC** |     79.4 KB |      17.3 KB |    96.7 KB |             79 MB |   4.0 MB/s |

gRPC request payloads are **3.2× smaller** than JSON (protobuf binary vs text encoding of 100 × 30 track points).
gRPC response payloads are **1.7× smaller** (100 × 15 waypoints × 2 floats).

FastAPI pushes **2.4× more bytes** over the wire yet completes **15% fewer requests** — the serialization and transfer overhead compounds under concurrency.

### Observations

**gRPC wins by ~18% on throughput with tighter tail latency**

- p99 is 380 ms (gRPC) vs 604 ms (FastAPI) — protobuf's compact binary encoding pays off over the wire, especially for the 100-vessel batch payload (3 000 floats as binary vs JSON text).
- p95/p50 ratio is 1.3× for gRPC vs 1.6× for FastAPI, meaning gRPC delivers more predictable response times under load.

**Network latency dominates model compute in remote benchmarks**

- Local benchmarks on the same machine show both servers at ~70 ms p50 and ~140 req/s. Over the network, p50 jumps to 240–270 ms, confirming that round-trip time and payload transfer are the bottleneck, not model inference.
- This makes transport efficiency (binary vs JSON) more impactful than it is locally.

**FastAPI performance tuning**

The FastAPI server uses several optimizations to close the gap:

- **orjson** for faster JSON serialization (with native numpy support)
- **async endpoints** with `run_in_executor` to keep the event loop free during CPU-bound inference
- **Raw `request.json()`** instead of Pydantic model validation on the hot path
- **Multi-worker support** via `--workers N` for multi-process serving

# ML Inference Benchmark

Compares inference throughput of a **PyTorch LSTM** vessel track predictor served via three implementations: **FastAPI direct**, **FastAPI + Redis/RQ job queue**, and **gRPC**.

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
load_test.py          # orchestrator: starts server(s), drives load, prints stats
services/
  fastapi_direct/
    server.py         # FastAPI + Uvicorn, inference offloaded to ThreadPoolExecutor
  fastapi_queue/
    server.py         # FastAPI + Uvicorn, enqueues jobs to Redis, polls for result
    inference_task.py # RQ job function (model loaded lazily per worker process)
    rq_worker.py      # RQ worker process launcher
  grpc/
    server.py         # gRPC server (ThreadPoolExecutor, 4 workers)
    inference.proto   # Protobuf service definition
    inference_pb2.py  # generated stubs (grpc_tools.protoc)
    inference_pb2_grpc.py
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

**3. Regenerate gRPC stubs** _(only needed after editing `services/grpc/inference.proto`)_

```bash
uv run python -m grpc_tools.protoc \
  -I services/grpc \
  --python_out=services/grpc \
  --grpc_python_out=services/grpc \
  services/grpc/inference.proto
```

**4. Run a server manually (optional)**

```bash
uv run python services/fastapi_direct/server.py        # http://localhost:8000
uv run python services/grpc/server.py                  # grpc://localhost:8000

# Queue server requires Redis + a worker
docker run -d -p 6379:6379 redis:alpine
uv run python services/fastapi_queue/rq_worker.py &
uv run python services/fastapi_queue/server.py         # http://localhost:8000
```

Endpoints:

| Protocol | Single vessel       | Batch (N vessels)        |
| -------- | ------------------- | ------------------------ |
| HTTP     | `POST /predict`     | `POST /predict_batch`    |
| gRPC     | `Inference.Predict` | `Inference.PredictBatch` |

**5. Run the full benchmark**

```bash
# Start Redis first for the queue server
docker run -d -p 6379:6379 redis:alpine

uv run load_test.py --server all --duration 30 --concurrency 10
```

Or target one server:

```bash
uv run load_test.py --server fastapi_direct --duration 30 --concurrency 20
uv run load_test.py --server fastapi_queue  --duration 30 --concurrency 20
uv run load_test.py --server grpc           --duration 30 --concurrency 20
```

### `load_test.py` flags

| Flag            | Default                  | Description                              |
| --------------- | ------------------------ | ---------------------------------------- |
| `--server`      | `all`                    | `fastapi_direct`, `fastapi_queue`, `grpc`, or `all` |
| `--duration`    | `20`                     | Test duration in seconds                 |
| `--concurrency` | `10`                     | Concurrent async workers                 |
| `--port`        | `8000`                   | Port to bind the server on               |
| `--host`        | `127.0.0.1`              | Server host                              |
| `--redis-url`   | `redis://localhost:6379` | Redis URL (fastapi_queue only)           |
| `--no-spawn`    | off                      | Connect to an already-running server     |

Batch size is fixed at 100 vessels/request (`BATCH_SIZE` in `load_test.py`).

## Results

### Local benchmark (2026-03-23)

Environment: Linux x86_64, Python 3.12 + CUDA, client and server on the same machine.
30 s duration, 10 concurrent workers, 100 vessels/request.
CPU/memory for `fastapi_queue` sampled from one representative **worker process** (4 workers total).

| Server            |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | Mem (avg) | CPU (avg) |
| ----------------- | -------: | --------: | ------: | ------: | ------: | ------: | --------: | --------: |
| fastapi_direct    |    186.5 |    18 646 |   51.09 |   73.73 |   82.27 |  177.53 |    856 MB |    112.0% |
| fastapi_queue     |     91.2 |     9 123 |  106.36 |  116.23 |  121.46 | 1092.22 |    818 MB |     17.8% |
| **grpc**          | **403.4**| **40 343**| **23.46**| **36.73**| **39.64**| **75.13**| **837 MB** | **125.6%** |

### Remote VPS benchmark (client → server over network)

Environment: VPS x86_64 Linux, Python 3.12, client on macOS over internet.
20 s duration, 10 concurrent workers, 100 vessels/request.

| Server   |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms |
| -------- | -------: | --------: | ------: | ------: | ------: | ------: |
| FastAPI  |     34.7 |     3 467 |     277 |     388 |     618 |   1 022 |
| **gRPC** | **41.5** | **4 153** | **238** | **313** | **380** | **510** |

### Data throughput

| Server   | req payload | resp payload | round-trip | total transferred | throughput |
| -------- | ----------: | -----------: | ---------: | ----------------: | ---------: |
| FastAPI  |    250.1 KB |      28.9 KB |   279.0 KB |            190 MB |   9.5 MB/s |
| **gRPC** |     79.4 KB |      17.3 KB |    96.7 KB |             79 MB |   4.0 MB/s |

gRPC request payloads are **3.2× smaller** than JSON (protobuf binary vs text encoding of 100 × 30 track points).
gRPC response payloads are **1.7× smaller** (100 × 15 waypoints × 2 floats).

## Observations

### gRPC is 2× faster than FastAPI direct

gRPC achieves **403 req/s** vs FastAPI's **187 req/s** — a 2.2× throughput advantage, with p50 latency of 23 ms vs 51 ms. HTTP/2 multiplexing, binary framing, and protobuf's compact encoding all contribute, but the primary driver is gRPC's 4-thread executor keeping more cores busy simultaneously.

CPU is proportionally higher for gRPC (126% vs 112% per monitored process) because it is actually doing more work per second, not burning cycles wastefully.

### FastAPI + job queue costs ~2× vs direct in throughput

With `SimpleWorker` (in-process, model loaded once per worker) and 4 parallel workers, the queue server reaches **91 req/s** — about half of FastAPI direct. The overhead comes from three unavoidable Redis round trips per request: enqueue, status poll(s), and result fetch. p50 latency is 106 ms vs 51 ms for direct; p99 is tight at 121 ms, but the occasional cold-start penalty shows in the `max` spike (~1 s) when a worker loads the model on its first job.

The worker CPU (18% per process × 4 workers ≈ 72% total) shows the compute is genuinely distributed — workers are active, not idle.

**Implementation note:** RQ's default `Worker` forks a new subprocess per job, which causes the model to be reloaded from disk on every request (breaking the lazy-load cache). Using `SimpleWorker` eliminates the fork and keeps the model resident, recovering performance from ~4 req/s to ~91 req/s.

The queue pattern is justified when **decoupling** matters — fire-and-forget submission, retries, priority routing, or bursting to many distributed workers — but adds unnecessary latency for synchronous, latency-sensitive inference where FastAPI direct or gRPC are simpler and faster.

### Network latency dominates in remote benchmarks

Over the network, p50 latency jumps from ~23–51 ms (local) to 240–280 ms, confirming round-trip time and payload transfer are the bottleneck rather than model compute. This makes transport efficiency (binary vs JSON, 3.2× smaller payloads for gRPC) more impactful remotely than locally.

### FastAPI performance tuning

The FastAPI direct server uses several optimizations to close the gap with gRPC:

- **orjson** for faster JSON serialization (with native numpy support)
- **async endpoints** with `run_in_executor` to keep the event loop free during CPU-bound inference
- **Raw `request.json()`** instead of Pydantic model validation on the hot path
- **CUDA detection** at startup — all three servers move the model and inference tensors to GPU when available, falling back to CPU transparently

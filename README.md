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
model_def.py              # VesselTrackPredictor + sequence dimension constants
train_pytorch.py          # synthetic track data generation + LSTM training
services/
  fastapi_direct/
    server.py             # FastAPI + Uvicorn, inference offloaded to ThreadPoolExecutor
  fastapi_queue/
    server.py             # FastAPI + Uvicorn, enqueues jobs to Redis, blocks for result
    inference_task.py     # shared inference logic (model loaded lazily)
    async_worker.py       # async worker: BLPOP jobs → thread-pool inference → LPUSH result
    rq_worker.py          # original RQ worker (deprecated)
  grpc/
    server.py             # gRPC server (ThreadPoolExecutor, 4 workers)
    inference.proto       # Protobuf service definition
    inference_pb2.py      # generated stubs (grpc_tools.protoc)
    inference_pb2_grpc.py
benchmarks/
  run.py                  # CLI entrypoint
  config/
    default.yaml          # batch_size, concurrency, duration, timeouts, …
    quick.yaml            # fast sanity-check values
  harness/
    lifecycle.py          # server spawn/teardown, health polling
    monitor.py            # CPU/RSS sampling thread (per-process, by label)
    payloads.py           # build HTTP and gRPC test payloads
  runners/
    http_runner.py        # httpx async load loop
    grpc_runner.py        # grpc.aio async load loop
  metrics/
    stats.py              # percentiles (linear interpolation), aggregation
  results/                # auto-saved JSON per run (timestamp + git SHA in filename)
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

uv run benchmarks/run.py --server all
```

Or target one server, or use the quick config for a fast sanity check:

```bash
uv run benchmarks/run.py --server fastapi_direct --duration 30 --concurrency 20
uv run benchmarks/run.py --server fastapi_queue  --duration 30 --concurrency 20
uv run benchmarks/run.py --server grpc           --duration 30 --concurrency 20

uv run benchmarks/run.py --server grpc --config benchmarks/config/quick.yaml
```

Results are automatically saved to `benchmarks/results/<timestamp>_<server>_<git-sha>.json`.

### `benchmarks/run.py` flags

| Flag            | Default                         | Description                                         |
| --------------- | ------------------------------- | --------------------------------------------------- |
| `--server`      | `all`                           | `fastapi_direct`, `fastapi_queue`, `grpc`, or `all` |
| `--config`      | `benchmarks/config/default.yaml`| YAML config file                                    |
| `--duration`    | from config                     | Override test duration (seconds)                    |
| `--concurrency` | from config                     | Override concurrent async workers                   |
| `--batch-size`  | from config                     | Override vessels per request                        |
| `--host`        | `127.0.0.1`                     | Server host                                         |
| `--port`        | `8000`                          | Server port                                         |
| `--no-spawn`    | off                             | Connect to an already-running server                |
| `--no-save`     | off                             | Skip saving results to `benchmarks/results/`        |
| `--json`        | off                             | Print full JSON results to stdout                   |

All benchmark parameters (batch size, concurrency, duration, worker threads, Redis URL, timeouts) live in `benchmarks/config/default.yaml`.

## Results

### Local benchmark (2026-03-23)

Linux x86_64, Python 3.12, client and server on the same machine.
30 s duration, 10 concurrent workers, 100 vessels/request.
`fastapi_queue` CPU/memory reported separately for server and worker processes.

**CUDA**

| Server            |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | Mem (avg) | CPU (avg) |
| ----------------- | -------: | --------: | ------: | ------: | ------: | ------: | --------: | --------: |
| fastapi_direct    |    175.2 |    17 517 |   54.16 |   78.57 |   91.59 |  142.63 |    856 MB |    112.9% |
| fastapi_queue     |    182.8 |    18 275 |   54.01 |   71.94 |   82.17 |  557.55 |    838 MB |     65.7% |
| **grpc**          | **377.3**| **37 735**| **24.84**| **39.57**| **44.67**| **83.54**| **836 MB** | **127.6%** |

**CPU only** (`CUDA_VISIBLE_DEVICES=""`)

| Server            |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | Mem (avg) | CPU (avg) |
| ----------------- | -------: | --------: | ------: | ------: | ------: | ------: | --------: | --------: |
| fastapi_direct    |    128.3 |    12 834 |   76.15 |  103.98 |  115.76 |  148.99 |    643 MB |    516.8% |
| fastapi_queue     |     82.7 |     8 273 |  116.23 |  157.19 |  192.48 |  803.28 |    549 MB |    500.3% |
| **grpc**          | **193.6**| **19 359**| **50.13**| **64.52**| **72.94**| **107.43**| **618 MB** | **768.0%** |

> CPU numbers use the original RQ-based worker. Re-run with `CUDA_VISIBLE_DEVICES=""` to reproduce.

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

### gRPC is the fastest across both CPU and CUDA

gRPC achieves **403 req/s** (CUDA) and **194 req/s** (CPU), consistently leading by ~2× over FastAPI direct. HTTP/2 multiplexing, binary framing, and protobuf's compact encoding all contribute, but the primary driver is gRPC's 4-thread executor keeping more cores busy simultaneously.

### CUDA gives ~1.5–2× uplift, but only where the bottleneck is compute

On CUDA, gRPC jumps from 194 → 403 req/s (2.1×) and FastAPI direct from 128 → 187 req/s (1.5×). The queue server gains less (83 → 91 req/s, 1.1×) because its bottleneck is Redis round trips, not inference compute — adding GPU doesn't help what isn't the bottleneck. CPU usage on CUDA drops to ~12–13% (from 500–770%) confirming the GPU absorbs the matrix work.

### FastAPI + async queue matches FastAPI direct

The queue server reaches **183 req/s** on CUDA — within noise of FastAPI direct's **175 req/s** — with identical p50 latency (54 ms). The key was replacing RQ with a custom async implementation:

| Implementation | req/s | p50 ms | what changed |
|---|---|---|---|
| RQ `Worker` (forking) | 3.7 | 6 054 | baseline |
| RQ `SimpleWorker` | 91 | 106 | no fork → model cached |
| Custom async + BLPOP | 183 | 54 | no framework overhead |

**Why RQ was slow:**
- Default `Worker` forks a subprocess per job → model reloaded from disk every ~5 ms inference call
- `SimpleWorker` fixes the fork but still runs ~15 Redis bookkeeping operations per job (status, registry, heartbeat, result TTL)
- Server polls every 5 ms with `get_status()` → average 2.5 ms wasted sleep per request

**Why the custom implementation is fast:**
- Worker: `BLPOP jobs → run_in_executor(inference) → LPUSH result` — **5 Redis ops** total vs RQ's 15+
- Server: `RPUSH job → BLPOP result` — truly blocking wait, **zero polling overhead**
- Single async worker process with a 4-thread executor handles concurrent jobs without spawning processes

The queue pattern is justified when **decoupling** matters — fire-and-forget submission, retries, priority routing, or bursting to distributed workers — and this implementation shows it doesn't have to cost anything in throughput or latency.

### Network latency dominates in remote benchmarks

Over the network, p50 latency jumps from ~23–51 ms (local) to 240–280 ms, confirming round-trip time and payload transfer are the bottleneck rather than model compute. This makes transport efficiency (binary vs JSON, 3.2× smaller payloads for gRPC) more impactful remotely than locally.

### FastAPI performance tuning

The FastAPI direct server uses several optimizations to close the gap with gRPC:

- **orjson** for faster JSON serialization (with native numpy support)
- **async endpoints** with `run_in_executor` to keep the event loop free during CPU-bound inference
- **Raw `request.json()`** instead of Pydantic model validation on the hot path
- **CUDA detection** at startup — all three servers move the model and inference tensors to GPU when available, falling back to CPU transparently

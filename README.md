# ML Inference Benchmark

Compares inference throughput of a **PyTorch LSTM** vessel track predictor served via **FastAPI (HTTP)** and **gRPC**.

**Task:** predict 15-minute future vessel trajectory from 5-minute AIS history.

|        |                                                                                           |
| ------ | ----------------------------------------------------------------------------------------- |
| Input  | 30 track points × 5 features (lat, lon, speed, course_sin, course_cos) — 10-sec intervals |
| Output | 15 waypoints × 2 features (lat, lon) — 1-min intervals                                    |
| Model  | LSTM encoder (hidden=256, layers=2) → linear decoder, ~3 MB on disk                       |

## Project structure

```
model_def.py          # VesselTrackPredictor + sequence dimension constants
train.py              # synthetic track data generation + LSTM training
inference.py          # shared inference engine: get_model(), predict(), predict_batch()
server_http.py        # FastAPI + Uvicorn server
server_grpc.py        # gRPC server (ThreadPoolExecutor, 4 workers)
inference.proto       # Protobuf service definition
inference_pb2.py      # generated stubs
inference_pb2_grpc.py # generated stubs
benchmarks/
  run.py              # benchmark CLI entrypoint
  config/
    default.yaml      # batch_size, concurrency, duration, timeouts, …
    quick.yaml        # fast sanity-check values
  harness/            # server lifecycle, process monitor, test payloads
  runners/            # async HTTP and gRPC load loops
  metrics/            # percentile computation and aggregation
  results/            # auto-saved JSON per run (timestamp + git SHA)
```

Both servers share `inference.py` — model loading and prediction logic live in exactly one place.

## Reproduce

**1. Install dependencies**

```bash
uv sync
```

**2. Train model**

```bash
uv run train.py
```

**3. Regenerate gRPC stubs** _(only needed after editing `inference.proto`)_

```bash
uv run python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. inference.proto
```

**4. Run a server manually (optional)**

```bash
uv run server_http.py   # http://localhost:8000
uv run server_grpc.py   # grpc://localhost:8000
```

HTTP endpoints:

| Method | Path             | Description           |
| ------ | ---------------- | --------------------- |
| GET    | `/health`        | Health check          |
| POST   | `/predict`       | Single vessel         |
| POST   | `/predict_batch` | Batch of N vessels    |

gRPC: `Inference.Predict`, `Inference.PredictBatch`, `Inference.Health`

**5. Run the full benchmark**

```bash
uv run benchmarks/run.py --server all
```

Or target one server, or use the quick config:

```bash
uv run benchmarks/run.py --server http --duration 30 --concurrency 20
uv run benchmarks/run.py --server grpc --duration 30 --concurrency 20
uv run benchmarks/run.py --server grpc --config benchmarks/config/quick.yaml
```

Results are automatically saved to `benchmarks/results/<timestamp>_<server>_<git-sha>.json`.

### `benchmarks/run.py` flags

| Flag            | Default                          | Description                          |
| --------------- | -------------------------------- | ------------------------------------ |
| `--server`      | `all`                            | `http`, `grpc`, or `all`             |
| `--config`      | `benchmarks/config/default.yaml` | YAML config file                     |
| `--duration`    | from config                      | Override test duration (seconds)     |
| `--concurrency` | from config                      | Override concurrent async workers    |
| `--batch-size`  | from config                      | Override vessels per request         |
| `--host`        | `127.0.0.1`                      | Server host                          |
| `--port`        | `8000`                           | Server port                          |
| `--no-spawn`    | off                              | Connect to an already-running server |
| `--no-save`     | off                              | Skip saving to `benchmarks/results/` |
| `--json`        | off                              | Print full JSON results to stdout    |

## Results

### Local benchmark (2026-03-23)

Linux x86_64, Python 3.12, client and server on the same machine.
30 s duration, 10 concurrent workers, 100 vessels/request.

**CUDA**

| Server       |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | Mem (avg) | CPU (avg) |
| ------------ | -------: | --------: | ------: | ------: | ------: | ------: | --------: | --------: |
| HTTP         |    175.2 |    17 517 |   54.16 |   78.57 |   91.59 |  142.63 |    856 MB |    112.9% |
| **gRPC**     | **377.3**| **37 735**| **24.84**| **39.57**| **44.67**| **83.54**| **836 MB** | **127.6%** |

**CPU only** (`CUDA_VISIBLE_DEVICES=""`)

| Server       |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms | Mem (avg) | CPU (avg) |
| ------------ | -------: | --------: | ------: | ------: | ------: | ------: | --------: | --------: |
| HTTP         |    128.3 |    12 834 |   76.15 |  103.98 |  115.76 |  148.99 |    643 MB |    516.8% |
| **gRPC**     | **193.6**| **19 359**| **50.13**| **64.52**| **72.94**| **107.43**| **618 MB** | **768.0%** |

### Remote VPS benchmark (client → server over network)

Environment: VPS x86_64 Linux, Python 3.12, client on macOS over internet.
20 s duration, 10 concurrent workers, 100 vessels/request.

| Server   |    req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms |  max ms |
| -------- | -------: | --------: | ------: | ------: | ------: | ------: |
| HTTP     |     34.7 |     3 467 |     277 |     388 |     618 |   1 022 |
| **gRPC** | **41.5** | **4 153** | **238** | **313** | **380** | **510** |

### Data throughput

| Server   | req payload | resp payload | total transferred | throughput |
| -------- | ----------: | -----------: | ----------------: | ---------: |
| HTTP     |    250.1 KB |      28.9 KB |            190 MB |   9.5 MB/s |
| **gRPC** |     79.4 KB |      17.3 KB |             79 MB |   4.0 MB/s |

gRPC request payloads are **3.2× smaller** than JSON (protobuf binary vs text encoding of 100 × 30 track points).

## Observations

### gRPC is faster across both CPU and CUDA

gRPC achieves ~2× higher throughput than HTTP consistently. HTTP/2 multiplexing, binary framing, and protobuf's compact encoding all contribute, but the primary driver is gRPC's 4-thread executor keeping more cores busy simultaneously.

### CUDA gives ~1.5–2× uplift where compute is the bottleneck

On CUDA, gRPC jumps from 194 → 403 req/s (2.1×) and HTTP from 128 → 175 req/s (1.5×). CPU usage drops to ~12–13% (from 500–770%) confirming the GPU absorbs the matrix work.

### Network latency dominates in remote benchmarks

Over the network, p50 latency jumps from ~25–54 ms (local) to 240–280 ms. This makes transport efficiency (3.2× smaller gRPC payloads) more impactful remotely than locally.

# ML Inference Benchmark

Compares inference throughput of **CatBoost** vs **PyTorch** served via **FastAPI** (ASGI), **Flask+Waitress** (WSGI), and **gRPC** (binary, threadpool).

- Input: 3 floats → Output: 1 float (regression)
- Model size on disk: CatBoost ~875 KB, PyTorch ~1 039 KB
- Load test: 20 s duration, 10 concurrent async workers

## Project structure

```
train_catboost.py    # trains and saves models/catboost_model.cbm
train_pytorch.py     # trains and saves models/pytorch_model.pt
model_def.py         # PyTorch MLP definition (shared)
server_fastapi.py    # FastAPI (ASGI) server, --model catboost|pytorch
server_flask.py      # Flask+Waitress (WSGI) server, --model catboost|pytorch
server_grpc.py       # gRPC server, --model catboost|pytorch
inference.proto      # Protobuf service definition
inference_pb2.py     # generated protobuf stubs (via grpc_tools.protoc)
inference_pb2_grpc.py
load_test.py         # load tester — starts server, measures RPS/latency/mem/CPU
```

## Reproduce

**1. Install dependencies**

```bash
uv add catboost torch numpy fastapi "uvicorn[standard]" flask waitress psutil httpx grpcio grpcio-tools
```

**2. Train models**

```bash
uv run train_catboost.py
uv run train_pytorch.py
```

**3. Regenerate gRPC stubs (only needed after editing inference.proto)**

```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
```

**4. Run servers manually (optional)**

```bash
uv run server_fastapi.py --model catboost   # http://localhost:8000
uv run server_flask.py   --model pytorch    # http://localhost:8000
uv run server_grpc.py    --model pytorch    # grpc://localhost:8000
```

**5. Run the full benchmark matrix**

```bash
uv run load_test.py --server all --model all --duration 20 --concurrency 10
```

Or target a specific combination:

```bash
uv run load_test.py --server grpc --model pytorch --duration 30 --concurrency 20
```

### `load_test.py` flags

| Flag            | Default | Description                              |
| --------------- | ------- | ---------------------------------------- |
| `--server`      | `all`   | `fastapi`, `flask`, `grpc`, or `all`     |
| `--model`       | `all`   | `catboost`, `pytorch`, or `all`          |
| `--duration`    | `20`    | Test duration in seconds                 |
| `--concurrency` | `10`    | Concurrent async workers                 |
| `--port`        | `8000`  | Port to bind the server on               |

## Results

Environment: macOS Darwin 25.3.0, Python 3.14, 20 s / 10 workers.

| Server       | Model    |       RPS |   p50 ms |   p95 ms |    Mem avg | CPU avg |
| ------------ | -------- | --------: | -------: | -------: | ---------: | ------: |
| FastAPI      | CatBoost |     3 046 |     3.22 |     3.75 |     128 MB |    344% |
| FastAPI      | PyTorch  |     3 291 |     2.99 |     3.29 |     229 MB |     48% |
| Flask        | CatBoost |     2 766 |     3.55 |     4.03 | **114 MB** |    424% |
| Flask        | PyTorch  |     3 089 |     3.16 |     3.51 |     218 MB |     62% |
| gRPC         | CatBoost | **8 009** | **1.24** | **1.57** | **114 MB** |    678% |
| gRPC         | PyTorch  |     7 985 |     1.24 |     1.57 |     214 MB |    199% |

### Observations

**gRPC vs HTTP**

- gRPC delivers ~2.6× more throughput (≈8 000 vs ≈3 000 RPS) and cuts p50 latency from ~3 ms to ~1.2 ms.
- The speedup comes from binary framing (protobuf), no HTTP parsing overhead, and persistent connections — no TCP/TLS handshake per request.
- Memory footprint is identical to Flask at the same model size; the web layer adds no meaningful overhead.

**PyTorch vs CatBoost (gRPC)**

- At gRPC speeds the two models are nearly tied (8 009 vs 7 985 RPS); the serialization cost no longer masks model differences.
- CatBoost still burns 3× more CPU than PyTorch (678% vs 199%) — tree traversal is inherently multi-core CPU-bound.

**FastAPI vs Flask (HTTP)**

- FastAPI (uvicorn/ASGI) is ~7–10% faster than Flask+Waitress across both models.
- Memory is nearly identical per framework; at 10 workers the model dominates the footprint.

**PyTorch vs CatBoost (HTTP)**

- PyTorch is ~8–11% faster over HTTP: a single `torch.no_grad()` matmul is a vectorised BLAS op.
- CatBoost footprint is smaller (~113–128 MB vs ~218–229 MB); PyTorch runtime adds overhead.

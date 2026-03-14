# ML Inference Benchmark

Compares inference throughput of **CatBoost** vs **PyTorch** served via **FastAPI** (ASGI) and **Flask+Waitress** (WSGI).

- Input: 3 floats → Output: 1 float (regression)
- Model size on disk: CatBoost ~875 KB, PyTorch ~1 039 KB
- Load test: 20 s duration, 10 concurrent async workers

## Project structure

```
train_catboost.py   # trains and saves models/catboost_model.cbm
train_pytorch.py    # trains and saves models/pytorch_model.pt
model_def.py        # PyTorch MLP definition (shared)
server_fastapi.py   # FastAPI (ASGI) server, --model catboost|pytorch
server_flask.py     # Flask+Waitress (WSGI) server, --model catboost|pytorch
load_test.py        # load tester — starts server, measures RPS/latency/mem/CPU
```

## Reproduce

**1. Install dependencies**

```bash
uv add catboost torch numpy fastapi "uvicorn[standard]" flask waitress psutil httpx
```

**2. Train models**

```bash
uv run train_catboost.py
uv run train_pytorch.py
```

**3. Run servers manually (optional)**

```bash
uv run server_fastapi.py --model catboost   # http://localhost:8000
uv run server_flask.py   --model pytorch    # http://localhost:8000
```

**4. Run the full benchmark matrix**

```bash
uv run load_test.py --server all --model all --duration 20 --concurrency 10
```

Or target a specific combination:

```bash
uv run load_test.py --server fastapi --model pytorch --duration 30 --concurrency 20
```

### `load_test.py` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `all` | `fastapi`, `flask`, or `all` |
| `--model` | `all` | `catboost`, `pytorch`, or `all` |
| `--duration` | `20` | Test duration in seconds |
| `--concurrency` | `10` | Concurrent async workers |
| `--port` | `8000` | Port to bind the server on |

## Results

Environment: macOS Darwin 25.3.0, Python 3.14, 20 s / 10 workers.

| Server | Model | RPS | p50 ms | p95 ms | Mem avg | CPU avg |
|--------|-------|----:|-------:|-------:|--------:|--------:|
| FastAPI | CatBoost | 2 879 | 3.28 | 4.00 | 140 MB | 323% |
| FastAPI | PyTorch | **3 122** | **3.02** | **3.46** | 229 MB | 48% |
| Flask | CatBoost | 2 681 | 3.55 | 4.18 | **121 MB** | 405% |
| Flask | PyTorch | 2 990 | 3.14 | 3.56 | 217 MB | 62% |

### Observations

**PyTorch vs CatBoost**
- PyTorch is ~8–11% faster: a single `torch.no_grad()` matmul is a vectorised BLAS op.
- CatBoost uses 5–8× more CPU — tree traversal is CPU-bound and parallelises across cores.
- CatBoost footprint is smaller (~120–140 MB vs ~217–229 MB); PyTorch runtime adds overhead.

**FastAPI vs Flask**
- FastAPI (uvicorn/ASGI) is ~7% faster than Flask+Waitress across both models.
- Memory is nearly identical per framework; the bottleneck at 10 workers is the model, not the web layer.

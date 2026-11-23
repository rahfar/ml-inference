# ML Inference Bench

**Question:** what is the most efficient way to put an HTTP API in front of an ML model, within one Python process?

The project separates the things that can matter and measures one at a time:

| Axis | Options | Where |
|---|---|---|
| **Execution strategy:** where and when `predict()` runs | `inline_loop`, `inline_threadpool`, `serial`, `batched` | `serving/strategies.py` |
| **Wire format:** how the array travels | `json`, `msgpack`, `protobuf`, `raw` float32 bytes | `serving/codecs.py` |
| **Model cost:** how expensive one prediction is | `lstm` (PyTorch, ms per call), `null` (no work: the serving overhead alone) | `models/` |

All models take the same input and give the same output: a 5-minute vessel track (30 steps × 5 features = 150 floats) in, a 15-minute forecast (15 × 2 = 30 floats) out.

## Strategies

| Strategy | What it does | Compared with the one above, it shows... |
|---|---|---|
| `inline_loop` | `predict()` directly in the async handler; the event loop is blocked meanwhile | the common mistake |
| `inline_threadpool` | `predict()` in the server's thread pool, as FastAPI does for a sync `def` handler | many predictions at once |
| `serial` | a queue feeding one inference thread, one request at a time | the effect of limiting concurrency |
| `batched` | like `serial`, but everything waiting goes into a single `predict()` call | the effect of batching |

## Experiments

Defined in `experiments.yaml`:

1. **`strategy`**: every strategy with the LSTM and JSON, at 1–64 concurrent users.
2. **`codec`**: every codec at 1, 32 and 512 samples per request, with the strategy fixed.

Each variant gets a fresh server process. Load is closed-loop (each virtual user sends its next request when the last one returns), spread over several client processes. Each level is a warmup, then a fixed measurement window. The server reports where it spent each request's time (`decode` / `wait` / `predict` / `encode`), and the results include those means. They also include the client's own CPU use, which flags runs where the client rather than the server was the bottleneck.

## Results

Not run yet on this structure.

## Usage

```bash
uv sync
make train                          # -> models/weights/
make profile MODEL=lstm             # model-only cost vs batch size
make bench EXP=strategy             # -> results/strategy.json + table
make bench EXP=codec
make serve ARGS="--strategy serial --codec raw"   # one server by hand, port 8000
```

## Layout

```
models/        data.py (synthetic tracks), lstm.py, null.py, train.py, profile.py
serving/       app.py (the server), strategies.py, codecs.py, tensor.proto
bench/         load.py (load generator), run.py (runs an experiment), report.py
experiments.yaml
```

## Limitations

- **Client and server share one machine** and talk over loopback. The client-CPU column shows when the client saturates, but not when it slows the server down.
- **Closed-loop load understates tail latency** under overload. An open-loop, fixed-arrival-rate mode would show it better.
- **Single runs** without repeats; expect roughly ±10% between runs.

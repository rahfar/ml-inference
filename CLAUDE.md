# CLAUDE.md

## Project

Benchmark of how to serve an ML model behind an HTTP API **in one Python process**. Axes, measured one at a time:
- execution strategy (`serving/strategies.py`): `inline_loop`, `inline_threadpool`, `serial`, `batched`
- wire format (`serving/codecs.py`): `json`, `msgpack`, `protobuf`, `raw`
- model cost (`models/`): `lstm` (PyTorch), `null` (no-op, overhead floor)

All models: `predict((n, 150) float32) -> (n, 30) float32`.

There is one server (`serving/app.py`, FastAPI + uvicorn); model, strategy and codec are CLI flags. It reports per-request `Server-Timing` (decode / wait / predict / encode) and `X-Batch-Size` headers, which the bench aggregates.

## Layout

```
models/          data.py, lstm.py, null.py, train.py, profile.py; weights/ (git-ignored)
serving/         app.py, strategies.py, codecs.py, tensor.proto + tensor_pb2.py (generated)
bench/           load.py (multi-process closed-loop client), run.py (experiment runner), report.py
experiments.yaml experiments = fixed settings + `vary` lists (cartesian product = variants)
results/         output + server logs (git-ignored)
```

## Commands

Package manager: `uv`. Everything runs locally.

```bash
uv sync
make train                        # required before serving the lstm
make profile MODEL=lstm
make bench EXP=strategy           # or: uv run python -m bench.run codec --duration 3 --levels 1 16
make serve ARGS="--strategy batched --codec raw"
make proto                        # regenerate serving/tensor_pb2.py (don't edit/format it)
make style
```

## Notes

- torch's thread count is per calling thread (OpenMP); `LSTMModel.predict` sets it in whichever thread runs it.
- `bench.run` spawns a fresh server per variant on port 8000 and refuses to start if the port is already served.
- No test framework. Lint/format with ruff (`make style`); exclude `serving/tensor_pb2.py`.

# CLAUDE.md

## Project

Benchmark of how to serve an ML model behind an HTTP API **in one Python process**. Axes, measured one at a time:
- execution strategy (`serving/strategies.py`): `inline_loop`, `inline_threadpool`, `serial`, `batched`
- wire format (`serving/codecs.py`): `json`, `msgpack`, `protobuf`, `raw`
- model cost (`models/`): `lstm` (PyTorch, ~0.6 ms/call), `catboost` (MultiRMSE, ~0.09 ms/call), `noop` (returns zeros: the serving overhead alone)

All models: `predict((n, 150) float32) -> (n, 30) float32`, absolute positions in and out; internally they work on offsets from the last observed position (`models/data.py`).

There is one server (`serving/app.py`, FastAPI + uvicorn); model, strategy and codec are CLI flags. It reports per-request `Server-Timing` (decode / wait / predict / encode) and `X-Batch-Size` headers, which the bench aggregates.

## Layout

```
models/          data.py, lstm.py, boosting.py (CatBoost), noop.py, train.py, profile.py; weights/ (git-ignored)
serving/         app.py, strategies.py, codecs.py, tensor.proto + tensor_pb2.py (generated)
bench/           load.py (multi-process closed-loop client), run.py (experiment runner), report.py
experiments.yaml experiments = fixed settings + `vary` lists (cartesian product = variants)
results/         output + server logs (git-ignored)
```

## Commands

Package manager: `uv`. Everything runs locally.

```bash
uv sync
make train                        # trains lstm + catboost (~1 min); `python -m models.train catboost` for one
make profile MODEL=catboost
make bench EXP=strategy           # or: uv run python -m bench.run codec --duration 3 --levels 1 16
make serve ARGS="--strategy batched --codec raw"
make proto                        # regenerate serving/tensor_pb2.py (don't edit/format it)
make style
```

## Notes

- `models/boosting.py` is not named `catboost.py` to avoid shadowing the `catboost` package.
- torch's thread count is per calling thread (OpenMP); `LSTMModel.predict` sets it in whichever thread runs it.
- `bench.run` spawns a fresh server per variant on port 8000 and refuses to start if the port is already served.
- No test framework. Lint/format with ruff (`make style`); exclude `serving/tensor_pb2.py`.

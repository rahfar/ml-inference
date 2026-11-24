# CLAUDE.md

## Project

Benchmark of how to serve an ML model behind an HTTP API **in one Python process**, plus a separate C++ server (`cpp/`) as the reference for what Python costs. Axes, measured one at a time:
- execution strategy (`serving/strategies.py`): `inline_loop`, `inline_threadpool`, `serial`, `batched` (`--workers N` inference threads, default 1)
- wire format (`serving/codecs.py`): `json`, `msgpack`, `protobuf`, `raw`
- model cost (`models/`): `lstm` (PyTorch, ~0.6 ms/call), `catboost` (MultiRMSE, ~0.09 ms/call), `noop` (returns zeros: the serving overhead alone)
- web stack (`serving/app.py`, `--stack`): fastapi|starlette on uvicorn|granian (ASGI), flask on gunicorn|granian (WSGI; inline_threadpool only)
- C++ vs Python (`cpp/server.cpp`): Drogon + libtorch (TorchScript) + libcatboostmodel + simdjson, same API and headers; experiment variants with `stack: cpp` run `cpp/build/server` directly

All models: `predict((n, 150) float32) -> (n, 30) float32`, absolute positions in and out; internally they work on offsets from the last observed position (`models/data.py`).

Request logic lives in `serving/handler.py` (framework-free: decode -> strategy -> encode, timed); `serving/app.py` has thin FastAPI/Starlette/Flask adapters and a CLI that sets `SERVING_*` env vars and execs the chosen server (uvicorn, granian, gunicorn) on an app factory. Replies carry `Server-Timing` (decode / wait / predict / encode) and `X-Batch-Size` headers, which the bench aggregates.

## Layout

```
models/          data.py, lstm.py, boosting.py (CatBoost), noop.py, train.py, profile.py; weights/ (git-ignored)
serving/         app.py, handler.py, strategies.py, codecs.py, tensor.proto + tensor_pb2.py/.pyi (generated)
bench/           load.py (multi-process closed-loop client), run.py (experiment runner), report.py
cpp/             server.cpp, export_lstm.py (-> models/weights/lstm.ts), Makefile; build/ + third_party/ git-ignored
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
make serve ARGS="--strategy batched --codec raw"   # or --stack flask+gunicorn --strategy inline_threadpool
make cpp                          # C++ server -> cpp/build/server (needs `brew install drogon simdjson`)
make proto                        # regenerate serving/tensor_pb2.py + .pyi stub (don't edit/format them)
make style                        # ruff autofix + format
make lint                         # check only: ruff lint, ruff format --check, ty
```

## Notes

- `models/boosting.py` is not named `catboost.py` to avoid shadowing the `catboost` package.
- torch's thread count is per calling thread (OpenMP); `LSTMModel.predict` sets it in whichever thread runs it.
- torch's intra-op threads barely speed up this LSTM (one call keeps ~1-2 cores busy); parallelism comes from several calls at once, so `batched` with `workers` ~ performance cores and `threads: 1` is the fast setup.
- `bench.run` spawns a fresh server per variant on port 8000 and refuses to start if the port is already served.
- No test framework. `make lint` must pass (ruff + ty); ruff skips the generated `serving/tensor_pb2.py(i)`.

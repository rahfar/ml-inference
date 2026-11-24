# ML Inference Bench

## Objective

**What is the most efficient way to put an HTTP API in front of an ML model, within one Python process? And how much of the cost is Python itself?**

"Efficient" means throughput (requests/s, or samples/s when requests carry several samples) and P99 latency at a given number of concurrent users, on one CPU machine. A separate C++ server, built to be as fast as possible, is the reference for what Python costs.

## Methodology

### What varies

Each experiment varies one or two of these and holds the rest fixed:

| Axis | Options | Where |
|---|---|---|
| **Execution strategy:** where and when `predict()` runs | `inline_loop`, `inline_threadpool`, `serial`, `batched` (see below) | `serving/strategies.py` |
| **Wire format:** how the array travels | `json`, `msgpack`, `protobuf`, `raw` float32 bytes | `serving/codecs.py` |
| **Model cost:** how expensive one prediction is | `lstm` (PyTorch, ~0.6 ms per call), `catboost` (~0.1 ms per call), `noop` (returns zeros: the serving overhead alone) | `models/` |
| **Web stack:** framework + HTTP server | FastAPI or Starlette (async, ASGI) on uvicorn or granian; Flask (sync, WSGI) on gunicorn or granian | `serving/app.py` |
| **Python vs C++:** the same server without the interpreter | Drogon + libtorch + CatBoost's C library + simdjson | `cpp/` |

| Strategy | What it does |
|---|---|
| `inline_loop` | `predict()` directly in the async handler; the event loop is blocked meanwhile |
| `inline_threadpool` | `predict()` in the server's thread pool, as FastAPI does for a sync `def` handler |
| `serial` | a queue feeding one inference thread, one request at a time |
| `batched` | a queue feeding N inference threads (`--workers`, default 1); each takes everything waiting, up to 64 rows, as one `predict()` call |

### Models

Both models solve the same task: a 5-minute vessel track (30 steps × 5 features = 150 floats) in, a 15-minute forecast (15 × 2 = 30 floats) out. They're trained on synthetic tracks (`make train`), and both beat the last-position baseline by more than 100× in validation MSE. What matters here is their cost: a call to the LSTM costs ~0.57 ms and gets 5× cheaper per sample in large batches; a call to CatBoost costs ~0.09 ms, mostly fixed overhead, and gets 40× cheaper per sample. Details in [results/models.md](results/models.md).

### Experiments

Defined in `experiments.yaml`:

| # | Experiment | Varies | Fixed |
|---|---|---|---|
| 1 | `strategy` | strategy × model, at 1–64 users | FastAPI + uvicorn, JSON, 1 sample/request, 4 torch threads |
| 1b | `batched_workers`, `threadpool_ref` | `batched`: workers (1, 4) × torch threads (1, 4); `inline_threadpool`: torch threads (1, 4). CatBoost and LSTM | as in 1 |
| 2 | `codec` | codec × model × samples per request (1, 32, 512), at 1 and 16 users | FastAPI + uvicorn, `batched` |
| 3 | `stack` | web stack × model, at 1–64 users | `inline_threadpool` (the only strategy a sync WSGI app can express), JSON |
| 4 | `cpp` | Python vs C++ × model × workers (1, 4), at 1–64 users | `batched`, 1 torch thread per worker, JSON |

### Measurement

- **One server process per variant,** started fresh and stopped afterwards, so no state carries over.
- **Closed-loop load:** each virtual user sends its next request when the last one returns, spread over 4 client processes. Each level is a 2 s warmup, then 10 s measured.
- **Server-side breakdown:** every reply carries a `Server-Timing` header (decode / wait / predict / encode) and the rows in its `predict()` call. The results include their means.
- **Client CPU is recorded,** to flag levels where the load generator, not the server, was the limit.

### Setup

- **Hardware:** Apple M5 (4 performance + 6 efficiency cores), macOS. Client and server on the same machine, over loopback.
- **Python:** 3.14.3, torch 2.15.0, CatBoost 1.2.10, FastAPI 0.165, uvicorn 0.54 (uvloop + httptools); experiment 3 adds Starlette 1.10, Flask 3.1.3, granian 2.8.9 and gunicorn 27.0.1.
- **C++:** Apple clang 21 (`-O3 -mcpu=native -flto`), Drogon 1.9.14, simdjson 5.2.0, libtorch 2.15.0 (from the same wheel), libcatboostmodel 1.2.10.

### Limitations

- **Single runs** without repeats; expect roughly ±10% between runs.
- **Client and server share one machine.** The client-CPU figure shows when the client saturates, but not when it slows the server down. The Python load generator tops out around 80k RPS, below the C++ server's capacity on cheap models.
- **Closed-loop load understates tail latency** under overload. An open-loop, fixed-arrival-rate mode would show it better.
- **Performance and efficiency cores.** macOS decides where threads run, which may favour or penalise thread-heavy strategies.

### Reproduce

```bash
uv sync
make train                          # lstm + catboost -> models/weights/ (~1 min)
make profile MODEL=lstm             # model-only cost vs batch size
make bench EXP=strategy             # -> results/strategy.json + table; likewise codec, stack, ...
make cpp                            # C++ server -> cpp/build/server (brew install drogon simdjson)
make bench EXP=cpp                  # Python vs C++
make serve ARGS="--strategy batched --workers 4 --threads 1"   # one server by hand, port 8000
```

## Results

Exact numbers, full tables and more observations: [results/](results/README.md).

### The short answer

**It depends on how expensive one prediction is, so profile the model first** (`make profile`).

- **Expensive model** (the LSTM, ~0.6 ms per call): get `predict()` off the event loop. Best is a batching queue with several inference workers and 1 torch thread each. A thread pool is the simple alternative. The wire format doesn't matter unless payloads are large.
- **Cheap model** (CatBoost, ~0.1 ms per call): the cost of each request dominates, so make fewer, bigger `predict()` calls. Batch on the server, use a binary format, and accept many samples per request.
- **Web stack:** any async stack is fine, and so is Flask on granian. Avoid gunicorn's `gthread` worker. Prefer async because it makes batching easy, not because it's faster per request.
- **C++** pays off for cheap models (~5× or more), much less for expensive ones (13–30%).

### 1. Strategy ([details](results/strategy.md))

- **Blocking the event loop caps an expensive model:** the LSTM on `inline_loop` stays at ~1,520 RPS whatever the load, with an 83 ms P99 at 64 users.
- **Batching wins for a cheap model:** CatBoost on `batched` reaches 12.9k RPS at 64 users, 2.3× any other strategy, with half the P99 or less.
- **An expensive model needs several inference workers to batch well.** torch's own threads barely parallelize this LSTM, so with one worker `batched` ran on about one core and lost to the thread pool. With 4 workers × 1 torch thread it reaches 7,960 RPS at 64 users, against 4,109 for the thread pool, and the P99 drops from 22 to 14.5 ms.
- **Every thread hop costs something:** about 25% of the `noop` ceiling (12.1k vs 16.0k RPS).

### 2. Codec ([details](results/codec.md))

- **For single samples, the format barely matters:** within 5% for the LSTM. JSON costs ~30% of throughput for `noop` and CatBoost at 16 users.
- **For large payloads, JSON is the bottleneck:** at 512 samples per request it spends ~13 ms per request on decoding and encoding, against 1.9 ms for msgpack, 0.25 ms for protobuf and ~0 for raw. CatBoost does 12× more samples/s with raw than with JSON.
- **Many samples per request is the biggest lever for a cheap model:** CatBoost with raw goes from 14.7k samples/s at 1 per request to 457k at 512 (31×). The LSTM gains 4.6×.

### 3. Web stack ([details](results/stack.md))

- **gunicorn's `gthread` worker is the bottleneck, not WSGI:** Flask on gunicorn is stuck at ~2,800 RPS even with `noop`, and halves the LSTM's throughput. The same Flask app on granian matches the async stacks.
- **FastAPI costs 15–20% over bare Starlette** at the `noop` ceiling, but only 2–4% with the LSTM. uvicorn and granian are close to even under load.
- **With a real model, the stack matters much less than the strategy:** the five non-gunicorn stacks are within ~10% of each other, while switching CatBoost from the thread pool to `batched` more than doubles its throughput.

### 4. Python vs C++ ([details](results/cpp.md))

- **Python costs ~0.15 ms per request before any model runs.** The C++ server spends ~0.01 ms of its own time per request.
- **Cheap model: C++ is 4.7× faster at 1 user and at least 5–6× under load** (CatBoost: 12.8k vs 2.7k RPS at 1 user; at least 73–79k vs 12.8k at 64 users, where the load generator, not the server, was the limit).
- **CatBoost's Python wrapper costs more than its trees:** ~80 µs per call, against 3 µs through CatBoost's C library. Most of it is building a training-oriented `Pool` object for every input ([breakdown](results/models.md#why-catboost-is-slow-from-python)).
- **Expensive model: C++ is only 13–30% faster** (LSTM: 10,449 vs 8,028 RPS at 64 users). Both run the same torch kernels, so C++ saves only the per-request overhead.

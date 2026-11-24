# ML Inference Bench

**Question:** what is the most efficient way to put an HTTP API in front of an ML model, within one Python process? A separate C++ server (`cpp/`) shows how much of the cost is Python itself.

The project separates the things that can matter and measures one at a time:

| Axis | Options | Where |
|---|---|---|
| **Execution strategy:** where and when `predict()` runs | `inline_loop`, `inline_threadpool`, `serial`, `batched` | `serving/strategies.py` |
| **Wire format:** how the array travels | `json`, `msgpack`, `protobuf`, `raw` float32 bytes | `serving/codecs.py` |
| **Model cost:** how expensive one prediction is | `lstm` (PyTorch, ~0.6 ms per call), `catboost` (~0.1 ms per call), `noop` (no work: the serving overhead alone) | `models/` |
| **Web stack:** framework + HTTP server | FastAPI or Starlette (async, ASGI) on uvicorn or granian; Flask (sync, WSGI) on gunicorn or granian | `serving/app.py` |
| **Python vs C++:** the same server without the interpreter | Drogon + libtorch + CatBoost's C library, batching on N workers | `cpp/` |

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

1. **`strategy`**: every strategy × every model, with JSON, at 1–64 concurrent users.
2. **`codec`**: every codec × every model at 1, 32 and 512 samples per request, with the strategy fixed to `batched`.
3. **`stack`**: every web stack × every model, with JSON and the strategy fixed to `inline_threadpool`, the only one a sync WSGI app can express.
4. **`cpp`**: Python (FastAPI + uvicorn) vs the C++ server × every model, both batching on 1 or 4 inference workers with 1 torch thread each, with JSON.

Each variant gets a fresh server process. Load is closed-loop (each virtual user sends its next request when the last one returns), spread over several client processes. Each level is a warmup, then a fixed measurement window. The server reports where it spent each request's time (`decode` / `wait` / `predict` / `encode`), and the results include those means. They also include the client's own CPU use, which flags runs where the client rather than the server was the bottleneck.

## Results

**Setup:** Apple M5 (4 performance + 6 efficiency cores), macOS, Python 3.14.3, torch 2.15.0, CatBoost 1.2.10, 4 threads per `predict()` call. Experiments 1 and 2 run FastAPI 0.165 on uvicorn 0.54 (uvloop + httptools); experiment 3 adds Starlette 1.10, Flask 3.1.3, granian 2.8.9 and gunicorn 27.0.1. Experiment 4 uses 1 thread per call, and the C++ server is built with Apple clang 21, Drogon 1.9.14, simdjson 5.2.0, libtorch 2.15.0 (from the same wheel) and libcatboostmodel 1.2.10. Closed-loop load from 4 client processes, 2 s warmup and 10 s measured per level, single runs (expect roughly ±10%). In experiments 1–3 the busiest client process never went above 44% of a core, so the numbers measure the server. In experiment 4 the C++ server outruns the client on cheap models (see there).

### The short answer

**It depends on how expensive one prediction is, so profile the model first** (`make profile`).

- **Expensive model** (the LSTM, ~0.6 ms per call): get `predict()` off the event loop. Best is a batching queue with several inference workers and 1 torch thread each (2× a thread pool at 64 users); a thread pool is the simple alternative. The wire format doesn't matter unless payloads are large.
- **Cheap model** (CatBoost, ~0.1 ms per call, mostly fixed overhead): the cost of each request dominates, so do fewer, bigger `predict()` calls. Batch on the server (`batched`: 2.3× throughput), use a binary format, and accept many samples per request. The last one is the biggest lever: 31× more samples/s at 512 samples per request than at 1.
- **Web stack:** any async stack (FastAPI or Starlette, on uvicorn or granian) is fine, and so is Flask on granian. With a real model they're within ~10% of each other. Avoid gunicorn's `gthread` worker: it caps a single process at ~2,800 RPS and halves the LSTM's throughput. Prefer async because it lets you batch, not because it's faster per request.
- **Python vs C++:** with the LSTM, an optimized C++ server running the same kernels is only 13–30% faster. With CatBoost or no model it's 5× faster or more, because Python's ~0.15 ms per request costs more than the model. Python CatBoost's own wrapper takes 84 µs per call; the C API takes 3 µs.

### The models

Both are trained on the same synthetic tracks (`make train`). Validation MSE is in degrees². The baseline predicts that the vessel stays where it is.

| Model | Size | Val MSE |
|---|---|---:|
| last-position baseline | – | 5.147 |
| `lstm`: 2-layer LSTM, hidden 256 | 803k parameters, 3.2 MB | 0.0302 |
| `catboost`: MultiRMSE, 300 trees, depth 6 | 4.8 MB | 0.0155 |

`predict()` cost with no HTTP involved (`make profile`, 4 threads):

| Batch | `lstm` ms/call | `lstm` µs/sample | `catboost` ms/call | `catboost` µs/sample |
|---:|---:|---:|---:|---:|
| 1 | 0.571 | 571 | 0.090 | 89.5 |
| 4 | 2.014 | 504 | 0.104 | 25.9 |
| 16 | 4.518 | 282 | 0.132 | 8.3 |
| 64 | 10.201 | 159 | 0.220 | 3.4 |
| 256 | 29.305 | 115 | 0.572 | 2.2 |

This table explains most of what follows. The LSTM's cost per sample falls 5× from batch 1 to batch 256, and a batch of 4 costs almost as much as 4 single calls. CatBoost's per-sample cost falls **40×**: a call is mostly fixed overhead.

### 1. Strategy: where `predict()` runs

JSON, 1 sample per request. `noop` returns zeros, so its row is the ceiling of the HTTP stack itself.

**Throughput (requests/s)**

| model | strategy | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| noop | inline_loop | 7,284 | 15,028 | 15,781 | 15,994 |
| noop | inline_threadpool | 4,843 | 9,190 | 11,389 | 12,128 |
| noop | serial | 5,647 | 10,283 | 12,299 | 13,304 |
| noop | batched | 4,812 | 10,141 | 13,839 | 15,052 |
| catboost | inline_loop | 3,571 | 5,487 | 5,595 | 5,650 |
| catboost | inline_threadpool | 2,694 | 4,512 | 5,450 | 5,713 |
| catboost | serial | 2,950 | 5,047 | 4,994 | 4,911 |
| catboost | batched | 2,687 | 5,944 | 10,291 | 12,896 |
| lstm | inline_loop | 1,300 | 1,520 | 1,523 | 1,516 |
| lstm | inline_threadpool | 1,183 | 3,540 | 3,948 | 4,108 |
| lstm | serial | 1,221 | 1,757 | 1,763 | 1,758 |
| lstm | batched | 1,174 | 1,541 | 2,544 | 4,371 |

**P99 latency (ms)**

| model | strategy | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| noop | inline_loop | 0.17 | 0.42 | 1.90 | 7.80 |
| noop | inline_threadpool | 0.23 | 1.00 | 2.03 | 5.79 |
| noop | serial | 0.20 | 0.52 | 1.50 | 5.29 |
| noop | batched | 0.23 | 0.83 | 1.41 | 6.57 |
| catboost | inline_loop | 0.30 | 1.17 | 5.45 | 22.22 |
| catboost | inline_threadpool | 0.40 | 2.15 | 4.05 | 13.63 |
| catboost | serial | 0.37 | 0.99 | 3.42 | 13.37 |
| catboost | batched | 0.40 | 1.81 | 2.17 | 6.62 |
| lstm | inline_loop | 0.80 | 4.02 | 19.78 | 82.55 |
| lstm | inline_threadpool | 0.88 | 1.45 | 6.71 | 22.07 |
| lstm | serial | 0.85 | 2.37 | 9.23 | 36.83 |
| lstm | batched | 0.88 | 2.73 | 6.79 | 22.66 |

- **Never block the event loop with an expensive model.** The LSTM on `inline_loop` is stuck at ~1,520 RPS whatever the load, and its P99 at 64 users (82.6 ms) is twice its median, because requests queue unfairly in front of a busy loop. For `noop` and CatBoost, `inline_loop` is fine at low load because `predict()` is short and there's no thread hop.
- **Every thread hop costs something.** At the `noop` ceiling, handing work to a thread pool costs about 25% (12.1k vs 16.0k RPS at 64 users).
- **The LSTM scales with threads; CatBoost doesn't.** torch releases the GIL, so `inline_threadpool` runs LSTM predictions in parallel: 3,540 RPS at 4 users vs 1,757 for `serial` (whose throughput is simply 1 / 0.57 ms). CatBoost on the thread pool stays at ~5.7k: concurrent calls contend, and each one slows from 0.11 to 0.65 ms.
- **Batching wins where calls are cheap.** CatBoost on `batched` reaches 12,896 RPS at 64 users, 2.3× any other strategy, with a P99 of 6.6 ms instead of 13–22 ms. Average batches are 9 rows at 16 users and 31 at 64.
- **For the LSTM, one batching thread only pays off at high load.** At 4 users the average batch of 2.5 rows saves nothing, because a batch of 4 costs almost 4 single calls. At 16 users, parallel threads beat one batching thread (3,948 vs 2,544 RPS). At 64 users they're even (4,371 vs 4,108 RPS, same P99). See the correction below: with several inference threads, batching wins.

**Correction, found after these runs: batching lost for the LSTM because it had only one inference thread.**
- **torch's own threads barely parallelize this LSTM.** One call with 4 torch threads keeps only 1–2 cores busy, and is no faster than with 1 thread up to batch 64.
- **So `batched` effectively ran on one core,** while the thread pool ran up to 40 predictions across every core.

With 4 inference threads pulling from the same queue (`--workers 4`) and 1 torch thread each, `batched` matches or beats the thread pool at every load. Measured separately with the same bench:

| LSTM, requests/s | users 1 | users 4 | users 16 | users 64 |
|---|---:|---:|---:|---:|
| `inline_threadpool` (1 torch thread) | 1,172 | 3,590 | 3,941 | 4,109 |
| `batched`, 1 worker × 4 torch threads (the table above) | 1,174 | 1,546 | 2,480 | 4,307 |
| `batched`, 4 workers × 1 torch thread | 1,170 | **3,762** | **4,728** | **7,960** |

- **At 64 users the tail drops too,** from 22 to 14.5 ms P99.
- **For CatBoost, one worker was already right.** More workers only split the queue into smaller batches (12.3k vs 12.9k RPS at 64 users).
- **The rule:** use enough inference workers to keep up with the requests. That's one for a cheap model, and about one per performance core for an expensive one. Keep 1 torch thread per worker.

<details>
<summary>Full table with the server-side time breakdown</summary>

Fixed: codec=json, threads=4, samples=1
Server-side ms are means per request; batch = rows per predict() call.

| model | strategy | users | RPS | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noop | inline_loop | 1 | 7284 | 0.13 | 0.17 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 44% |
| noop | inline_loop | 4 | 15028 | 0.26 | 0.42 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 26% |
| noop | inline_loop | 16 | 15781 | 0.91 | 1.90 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 26% |
| noop | inline_loop | 64 | 15994 | 3.90 | 7.80 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 25% |
| noop | inline_threadpool | 1 | 4843 | 0.20 | 0.23 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 29% |
| noop | inline_threadpool | 4 | 9190 | 0.41 | 1.00 | 0.02 | 0.21 | 0.00 | 0.00 | 1.0 | 0 | 17% |
| noop | inline_threadpool | 16 | 11389 | 1.36 | 2.03 | 0.02 | 0.82 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | inline_threadpool | 64 | 12128 | 5.20 | 5.79 | 0.02 | 3.77 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | serial | 1 | 5647 | 0.18 | 0.20 | 0.02 | 0.03 | 0.00 | 0.00 | 1.0 | 0 | 35% |
| noop | serial | 4 | 10283 | 0.39 | 0.52 | 0.02 | 0.16 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | serial | 16 | 12299 | 1.30 | 1.50 | 0.02 | 0.65 | 0.00 | 0.00 | 1.0 | 0 | 19% |
| noop | serial | 64 | 13304 | 4.80 | 5.29 | 0.02 | 2.49 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | batched | 1 | 4812 | 0.21 | 0.23 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 30% |
| noop | batched | 4 | 10141 | 0.38 | 0.83 | 0.02 | 0.19 | 0.00 | 0.00 | 2.5 | 0 | 18% |
| noop | batched | 16 | 13839 | 1.15 | 1.41 | 0.02 | 0.60 | 0.00 | 0.00 | 10.3 | 0 | 21% |
| noop | batched | 64 | 15052 | 4.14 | 6.57 | 0.02 | 2.52 | 0.00 | 0.00 | 43.4 | 0 | 20% |
| catboost | inline_loop | 1 | 3571 | 0.28 | 0.30 | 0.02 | 0.00 | 0.11 | 0.01 | 1.0 | 0 | 23% |
| catboost | inline_loop | 4 | 5487 | 0.55 | 1.17 | 0.02 | 0.00 | 0.11 | 0.01 | 1.0 | 0 | 12% |
| catboost | inline_loop | 16 | 5595 | 2.67 | 5.45 | 0.02 | 0.00 | 0.10 | 0.01 | 1.0 | 0 | 10% |
| catboost | inline_loop | 64 | 5650 | 11.14 | 22.22 | 0.02 | 0.00 | 0.10 | 0.01 | 1.0 | 0 | 10% |
| catboost | inline_threadpool | 1 | 2694 | 0.37 | 0.40 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 18% |
| catboost | inline_threadpool | 4 | 4512 | 0.85 | 2.15 | 0.02 | 0.36 | 0.23 | 0.01 | 1.0 | 0 | 9% |
| catboost | inline_threadpool | 16 | 5450 | 2.86 | 4.05 | 0.02 | 1.65 | 0.56 | 0.01 | 1.0 | 0 | 9% |
| catboost | inline_threadpool | 64 | 5713 | 11.05 | 13.63 | 0.02 | 7.83 | 0.65 | 0.01 | 1.0 | 0 | 9% |
| catboost | serial | 1 | 2950 | 0.34 | 0.37 | 0.02 | 0.04 | 0.11 | 0.01 | 1.0 | 0 | 20% |
| catboost | serial | 4 | 5047 | 0.79 | 0.99 | 0.02 | 0.35 | 0.19 | 0.01 | 1.0 | 0 | 10% |
| catboost | serial | 16 | 4994 | 3.20 | 3.42 | 0.02 | 2.76 | 0.20 | 0.01 | 1.0 | 0 | 9% |
| catboost | serial | 64 | 4911 | 13.02 | 13.37 | 0.02 | 12.58 | 0.20 | 0.01 | 1.0 | 0 | 9% |
| catboost | batched | 1 | 2687 | 0.37 | 0.40 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 19% |
| catboost | batched | 4 | 5944 | 0.65 | 1.81 | 0.02 | 0.25 | 0.15 | 0.01 | 2.5 | 0 | 11% |
| catboost | batched | 16 | 10291 | 1.48 | 2.17 | 0.02 | 0.82 | 0.27 | 0.01 | 9.0 | 0 | 17% |
| catboost | batched | 64 | 12896 | 4.75 | 6.62 | 0.02 | 2.77 | 0.93 | 0.01 | 30.9 | 0 | 19% |
| lstm | inline_loop | 1 | 1300 | 0.77 | 0.80 | 0.02 | 0.00 | 0.57 | 0.01 | 1.0 | 0 | 9% |
| lstm | inline_loop | 4 | 1520 | 1.98 | 4.02 | 0.02 | 0.00 | 0.57 | 0.01 | 1.0 | 0 | 4% |
| lstm | inline_loop | 16 | 1523 | 9.85 | 19.78 | 0.02 | 0.00 | 0.57 | 0.01 | 1.0 | 0 | 3% |
| lstm | inline_loop | 64 | 1516 | 41.32 | 82.55 | 0.02 | 0.00 | 0.57 | 0.01 | 1.0 | 0 | 4% |
| lstm | inline_threadpool | 1 | 1183 | 0.84 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | inline_threadpool | 4 | 3540 | 1.10 | 1.45 | 0.03 | 0.15 | 0.76 | 0.01 | 1.0 | 0 | 8% |
| lstm | inline_threadpool | 16 | 3948 | 3.93 | 6.71 | 0.03 | 1.44 | 1.86 | 0.01 | 1.0 | 0 | 12% |
| lstm | inline_threadpool | 64 | 4108 | 15.41 | 22.07 | 0.02 | 9.77 | 2.76 | 0.01 | 1.0 | 0 | 11% |
| lstm | serial | 1 | 1221 | 0.82 | 0.85 | 0.02 | 0.04 | 0.57 | 0.01 | 1.0 | 0 | 9% |
| lstm | serial | 4 | 1757 | 2.27 | 2.37 | 0.03 | 1.47 | 0.57 | 0.01 | 1.0 | 0 | 4% |
| lstm | serial | 16 | 1763 | 9.07 | 9.23 | 0.03 | 8.27 | 0.56 | 0.01 | 1.0 | 0 | 4% |
| lstm | serial | 64 | 1758 | 36.39 | 36.83 | 0.03 | 35.60 | 0.56 | 0.01 | 1.0 | 0 | 4% |
| lstm | batched | 1 | 1174 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | batched | 4 | 1541 | 2.58 | 2.73 | 0.03 | 0.75 | 1.60 | 0.01 | 2.5 | 0 | 3% |
| lstm | batched | 16 | 2544 | 6.26 | 6.79 | 0.02 | 1.67 | 4.07 | 0.01 | 12.5 | 0 | 5% |
| lstm | batched | 64 | 4371 | 14.28 | 22.66 | 0.02 | 3.54 | 9.08 | 0.01 | 48.5 | 0 | 6% |

</details>

### 2. Codec: how the array travels

Strategy fixed to `batched`, the overall winner of experiment 1. Samples per request: 1, 32 and 512.

**Throughput (samples/s = requests/s × samples per request)**

| model | codec | samples 1, users 1 | samples 1, users 16 | samples 32, users 1 | samples 32, users 16 | samples 512, users 1 | samples 512, users 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| noop | json | 4,822 | 13,909 | 33,312 | 43,622 | 44,134 | 48,230 |
| noop | msgpack | 5,490 | 19,698 | 86,541 | 169,542 | 195,942 | 250,931 |
| noop | protobuf | 5,445 | 19,990 | 154,979 | 340,461 | 930,099 | 1,309,696 |
| noop | raw | 5,692 | 21,448 | 171,821 | 462,605 | 2,105,242 | 3,602,790 |
| catboost | json | 2,687 | 10,282 | 23,334 | 32,378 | 30,515 | 37,581 |
| catboost | msgpack | 2,994 | 13,866 | 58,621 | 123,040 | 138,598 | 194,202 |
| catboost | protobuf | 2,928 | 14,086 | 76,221 | 182,298 | 314,266 | 380,211 |
| catboost | raw | 3,013 | 14,669 | 86,675 | 202,778 | 379,699 | 456,858 |
| lstm | json | 1,174 | 2,546 | 4,525 | 5,658 | 8,499 | 10,086 |
| lstm | msgpack | 1,220 | 2,549 | 5,130 | 6,150 | 10,752 | 10,803 |
| lstm | protobuf | 1,229 | 2,531 | 5,254 | 6,195 | 11,315 | 11,418 |
| lstm | raw | 1,232 | 2,461 | 5,254 | 6,176 | 11,469 | 11,469 |

**Server-side decode + encode time per request (ms)**

| model | codec | samples 1, users 1 | samples 1, users 16 | samples 32, users 1 | samples 32, users 16 | samples 512, users 1 | samples 512, users 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| noop | json | 0.03 | 0.02 | 0.65 | 0.65 | 10.22 | 10.29 |
| noop | msgpack | 0.01 | 0.00 | 0.11 | 0.12 | 1.84 | 1.84 |
| noop | protobuf | 0.00 | 0.00 | 0.02 | 0.02 | 0.24 | 0.25 |
| noop | raw | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| catboost | json | 0.03 | 0.03 | 0.81 | 0.82 | 12.92 | 13.07 |
| catboost | msgpack | 0.01 | 0.00 | 0.12 | 0.12 | 1.86 | 1.82 |
| catboost | protobuf | 0.00 | 0.00 | 0.02 | 0.02 | 0.26 | 0.25 |
| catboost | raw | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lstm | json | 0.03 | 0.03 | 0.81 | 0.90 | 12.84 | 13.46 |
| lstm | msgpack | 0.01 | 0.01 | 0.12 | 0.13 | 1.90 | 2.20 |
| lstm | protobuf | 0.01 | 0.00 | 0.02 | 0.02 | 0.26 | 0.27 |
| lstm | raw | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |

- **For single samples, the format hardly matters for the LSTM** (within 5%). For `noop` and CatBoost, JSON costs about 30% of throughput at 16 users (13.9k vs 21.4k, and 10.3k vs 14.7k samples/s). Server-side decoding and encoding take only 0.03 ms, so most of that cost lies elsewhere. JSON bodies are 5× larger, and the client parses JSON too.
- **For large payloads, JSON becomes the bottleneck.** At 512 samples per request, decoding and encoding take ~13 ms per request in JSON, against 1.9 ms in msgpack, 0.25 ms in protobuf and ~0 in raw. For CatBoost that's a 12× throughput difference between JSON and raw (37.6k vs 457k samples/s). For the LSTM the model still dominates (10.1k vs 11.5k).
- **msgpack sits in between.** It's binary, but still a list of Python floats packed one by one. Protobuf's packed repeated float is a single buffer, close to raw. So the serialization part of gRPC's speed is protobuf's packed arrays; its HTTP/2 transport wasn't measured here.
- **Sending many samples per request is the biggest lever for a cheap model.** CatBoost with raw goes from 14.7k samples/s at 1 per request to 457k at 512 per request (31×). The LSTM goes from 2.5k to 11.5k (4.6×), as the profile predicts.
- **P99 with large payloads is queueing, not the codec.** At 512 samples and 16 users, each request waits behind ~15 others, so the LSTM's median is already 710–810 ms. That's the throughput limit showing as latency.

### 3. Web stack: framework + HTTP server

Strategy fixed to `inline_threadpool`: in every stack, `predict()` runs in one of 40 request threads. That's the only strategy a sync WSGI app can express. JSON, 1 sample per request.

**Throughput (requests/s)**

| model | stack | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 5,074 | 9,878 | 11,371 | 12,089 |
| noop | starlette+uvicorn | 5,312 | 11,883 | 13,178 | 14,771 |
| noop | fastapi+granian | 3,441 | 8,220 | 10,857 | 12,426 |
| noop | starlette+granian | 4,252 | 9,977 | 12,838 | 15,584 |
| noop | flask+gunicorn | 2,857 | 2,703 | 2,774 | 2,812 |
| noop | flask+granian | 6,725 | 12,636 | 12,094 | 15,541 |
| catboost | fastapi+uvicorn | 2,688 | 4,676 | 5,466 | 5,785 |
| catboost | starlette+uvicorn | 2,820 | 5,127 | 5,714 | 6,263 |
| catboost | fastapi+granian | 2,467 | 4,457 | 5,266 | 5,748 |
| catboost | starlette+granian | 2,594 | 4,899 | 5,734 | 6,313 |
| catboost | flask+gunicorn | 2,127 | 2,111 | 2,166 | 2,167 |
| catboost | flask+granian | 3,253 | 5,663 | 5,629 | 5,757 |
| lstm | fastapi+uvicorn | 1,172 | 3,593 | 3,899 | 4,065 |
| lstm | starlette+uvicorn | 1,195 | 3,752 | 4,084 | 4,240 |
| lstm | fastapi+granian | 1,104 | 3,492 | 3,891 | 4,038 |
| lstm | starlette+granian | 1,124 | 3,666 | 4,193 | 4,145 |
| lstm | flask+gunicorn | 1,060 | 2,043 | 1,986 | 1,997 |
| lstm | flask+granian | 1,253 | 4,047 | 3,949 | 3,773 |

**P99 latency (ms)**

| model | stack | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 0.23 | 0.49 | 2.06 | 6.23 |
| noop | starlette+uvicorn | 0.21 | 0.40 | 2.82 | 5.24 |
| noop | fastapi+granian | 0.32 | 0.60 | 2.16 | 8.33 |
| noop | starlette+granian | 0.25 | 0.50 | 3.08 | 6.42 |
| noop | flask+gunicorn | 0.38 | 2.45 | 9.61 | 29.03 |
| noop | flask+granian | 0.17 | 1.88 | 6.29 | 74.51 |
| catboost | fastapi+uvicorn | 0.40 | 1.13 | 4.01 | 13.02 |
| catboost | starlette+uvicorn | 0.38 | 1.04 | 6.47 | 12.16 |
| catboost | fastapi+granian | 0.44 | 1.15 | 4.43 | 17.32 |
| catboost | starlette+granian | 0.42 | 1.05 | 4.09 | 15.34 |
| catboost | flask+gunicorn | 0.50 | 3.07 | 12.17 | 38.38 |
| catboost | flask+granian | 0.34 | 1.03 | 5.58 | 37.69 |
| lstm | fastapi+uvicorn | 0.88 | 1.41 | 6.72 | 22.11 |
| lstm | starlette+uvicorn | 0.87 | 1.34 | 6.50 | 39.15 |
| lstm | fastapi+granian | 0.94 | 1.40 | 6.23 | 25.33 |
| lstm | starlette+granian | 0.92 | 1.30 | 5.79 | 48.75 |
| lstm | flask+gunicorn | 0.98 | 2.83 | 13.15 | 46.35 |
| lstm | flask+granian | 0.83 | 1.19 | 9.20 | 39.88 |

- **gunicorn's thread worker is the bottleneck, not WSGI.** Flask on gunicorn (`gthread`) is stuck at ~2,800 RPS even with `noop`, and halves the LSTM's throughput (~2,000 vs ~4,000 RPS). The same Flask app on granian matches the async stacks, and is the fastest of all at 1 user (6,725 RPS on `noop`, 1,253 on the LSTM): granian parses HTTP in Rust and hands each request straight to a Python thread, without an event loop in between. Its weak spot is the tail at 64 users (P99 74.5 ms on `noop`), where 40 Python threads fight over the GIL.
- **FastAPI costs 15–20% over bare Starlette at the ceiling** (`noop` at 64 users: 12.1k vs 14.8k on uvicorn, 12.4k vs 15.6k on granian), but only 5–10% with CatBoost and 2–4% with the LSTM. The request handler here is the same in both, so that's FastAPI's own routing and request wrapping.
- **uvicorn vs granian is close to even under load.** On ASGI, granian is slower at 1 user (3.4–4.3k vs 5.1–5.3k RPS on `noop`; 6–8% slower with CatBoost and the LSTM) and level or slightly faster at 64 users.
- **With a real model, the stack matters much less than the strategy.** Across the five non-gunicorn stacks, CatBoost lands at 5.7–6.3k RPS and the LSTM at 3.8–4.2k at 64 users. Switching CatBoost from this thread-pool strategy to `batched` (experiment 1) gave 12.9k. Batching needs an async server, so that's the real argument for async: it makes batching easy. Raw speed isn't.

### 4. Python vs C++

`cpp/server.cpp` is a separate C++ server with the same API, models and timing headers, measured with the same load generator. It's built to be as fast as possible:
- **HTTP:** Drogon, an async framework with one event loop per I/O thread (2 threads).
- **LSTM:** libtorch from the same torch wheel, loading a TorchScript export. That means the same ATen kernels as Python.
- **CatBoost:** CatBoost's own C evaluation library.
- **JSON:** simdjson.
- **Build:** `-O3 -mcpu=native -flto`.

Both servers use the batching setup from the correction above: N inference workers, 1 torch thread each. JSON, 1 sample per request. Details in `cpp/README.md`.

**Model-only cost** (`make profile` vs `cpp/build/server --profile`, 1 thread, ms per call):

| Batch | `lstm` Python | `lstm` C++ | `catboost` Python | `catboost` C++ |
|---:|---:|---:|---:|---:|
| 1 | 0.573 | 0.554 | 0.084 | **0.003** |
| 4 | 2.040 | 2.022 | 0.082 | 0.011 |
| 16 | 3.569 | 3.530 | 0.109 | 0.027 |
| 64 | 9.946 | 9.886 | 0.194 | 0.088 |
| 256 | 36.697 | 36.527 | 0.557 | 0.334 |

**Throughput (requests/s)**

| model | server | workers | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| noop | Python (FastAPI + uvicorn) | 1 | 4,732 | 10,205 | 13,842 | 14,917 |
| noop | Python (FastAPI + uvicorn) | 4 | 4,766 | 9,866 | 14,001 | 14,417 |
| noop | C++ | 1 | 14,085 | 37,395 | 68,582 | 79,667 |
| noop | C++ | 4 | 14,504 | 37,060 | 67,602 | 78,292 |
| catboost | Python (FastAPI + uvicorn) | 1 | 2,704 | 6,628 | 10,238 | 12,836 |
| catboost | Python (FastAPI + uvicorn) | 4 | 2,753 | 5,366 | 8,830 | 12,055 |
| catboost | C++ | 1 | 12,843 | 35,090 | 62,352 | 73,387 |
| catboost | C++ | 4 | 12,540 | 34,848 | 63,757 | 78,920 |
| lstm | Python (FastAPI + uvicorn) | 1 | 1,171 | 1,559 | 3,116 | 4,747 |
| lstm | Python (FastAPI + uvicorn) | 4 | 1,129 | 3,758 | 4,740 | 8,028 |
| lstm | C++ | 1 | 1,479 | 1,648 | 3,218 | 5,497 |
| lstm | C++ | 4 | 1,476 | 4,362 | 5,353 | 10,449 |

**P99 latency (ms)**

| model | server | workers | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| noop | Python (FastAPI + uvicorn) | 1 | 0.24 | 0.48 | 1.42 | 7.42 |
| noop | Python (FastAPI + uvicorn) | 4 | 0.23 | 0.46 | 1.37 | 5.26 |
| noop | C++ | 1 | 0.13 | 0.17 | 0.60 | 2.29 |
| noop | C++ | 4 | 0.08 | 0.17 | 0.60 | 2.31 |
| catboost | Python (FastAPI + uvicorn) | 1 | 0.39 | 0.64 | 3.72 | 6.74 |
| catboost | Python (FastAPI + uvicorn) | 4 | 0.39 | 1.04 | 2.37 | 10.49 |
| catboost | C++ | 1 | 0.09 | 0.17 | 0.58 | 3.03 |
| catboost | C++ | 4 | 0.09 | 0.18 | 0.59 | 2.29 |
| lstm | Python (FastAPI + uvicorn) | 1 | 0.88 | 2.70 | 5.84 | 21.86 |
| lstm | Python (FastAPI + uvicorn) | 4 | 1.95 | 1.23 | 4.48 | 14.19 |
| lstm | C++ | 1 | 0.70 | 2.48 | 5.11 | 12.02 |
| lstm | C++ | 4 | 0.70 | 2.11 | 8.53 | 12.45 |

**C++ client-bound levels.** At 16 and 64 users, the C++ `noop` and `catboost` rows are **client-bound**: the Python load generator's processes are at 91–98% CPU. So those numbers are lower bounds on what the server can do. With 6 client processes the same server reached 87k RPS on `noop`, and with 8 it dropped again, because client and server had run out of cores. The Python rows and the LSTM rows used at most 30% client CPU.

- **Python costs ~0.15 ms per request before any model runs.** With `noop` at 1 user, Python does 4.7k RPS (P50 0.21 ms) and C++ 14.1k (P50 0.07 ms). Most of C++'s 0.07 ms is the Python client itself: the server reports ~0.01 ms of its own time per request.
- **Cheap model: C++ is 4.7× faster at 1 user and at least 5–6× under load.** CatBoost does 12.8k vs 2.7k RPS at 1 user, and at least 73–79k vs 12.8k at 64 users.
  - About a third of the 1-user gap (0.1 of 0.29 ms) is CatBoost's own Python wrapper: 84 µs per call, against **3 µs** for the same model through the C API. The wrapper costs more than the trees do.
  - Python's batching hides that cost under load (30 rows per call at 64 users). The C++ server barely needs to batch (2–11 rows).
- **Expensive model: C++ is 13–30% faster.** The LSTM runs the same kernels, and `predict()` takes the same ~5.0 ms per ~16-row batch in both servers at 64 users. C++ wins only on per-request overhead and queueing:
  - 1,476 vs 1,129 RPS at 1 user (P50 0.68 vs 0.85 ms).
  - 10,449 vs 8,028 at 64 users, with a P99 of 12.5 vs 14.2 ms.
- **One C++ weak spot: the LSTM tail at 4–16 users.** With 4 workers the C++ P99 is higher than Python's (2.1 vs 1.2 ms at 4 users, 8.5 vs 4.5 ms at 16), although its median is lower. I haven't investigated this, and it's a single run. A worker scheduled on an efficiency core would explain it.
- **When C++ pays off: it depends on the per-request overhead relative to the model's cost.**
  - **Expensive model (milliseconds per call):** Python with batching and several workers gets within 25% of an optimized C++ server.
  - **Cheap model (microseconds per call):** Python's ~0.15 ms overhead per request dominates, and C++ is 5× faster or more. Within Python, the remedy is fewer, bigger requests (experiment 2).

## Usage

```bash
uv sync
make train                          # lstm + catboost -> models/weights/ (~1 min)
make profile MODEL=lstm             # model-only cost vs batch size
make bench EXP=strategy             # -> results/strategy.json + table
make bench EXP=codec
make serve ARGS="--strategy serial --codec raw"   # one server by hand, port 8000
make serve ARGS="--stack flask+gunicorn --strategy inline_threadpool"
make cpp                            # C++ server -> cpp/build/server (brew install drogon simdjson)
make bench EXP=cpp                  # Python vs C++
```

## Layout

```
models/        data.py (synthetic tracks), lstm.py, boosting.py (CatBoost), noop.py, train.py, profile.py
serving/       app.py (framework adapters + server launch), handler.py (decode -> predict -> encode),
               strategies.py, codecs.py, tensor.proto
bench/         load.py (load generator), run.py (runs an experiment), report.py
cpp/           server.cpp (the C++ server), export_lstm.py (TorchScript export), Makefile
experiments.yaml
```

## Limitations

- **Client and server share one machine** and talk over loopback. The client-CPU column shows when the client saturates, but not when it slows the server down. The Python load generator tops out around 80k RPS, below the C++ server's capacity on cheap models.
- **Closed-loop load understates tail latency** under overload. An open-loop, fixed-arrival-rate mode would show it better.
- **Single runs** without repeats; expect roughly ±10% between runs. A few P99 cells are single-run outliers (e.g. codec: JSON at 32 samples, 16 users).
- **One machine with performance and efficiency cores.** macOS decides where threads run, which may favour or penalise thread-heavy strategies.

# 4. Python vs C++

Python (FastAPI on uvicorn) vs the C++ server (`cpp/server.cpp`) × every model. Both servers batch on 1 or 4 inference workers with 1 torch thread each, the best setup from the [workers follow-up](strategy.md#follow-up-batching-with-several-inference-workers). JSON, 1 sample per request.

The C++ server has the same API, models and timing headers, and is built to be as fast as possible:
- **HTTP:** Drogon, an async framework with one event loop per I/O thread (2 threads).
- **LSTM:** libtorch from the same torch wheel, loading a TorchScript export. That means the same ATen kernels as Python.
- **CatBoost:** CatBoost's own C evaluation library.
- **JSON:** simdjson.
- **Build:** `-O3 -mcpu=native -flto`.

Details in [`cpp/README.md`](../cpp/README.md). The model-only cost of both is in [models.md](models.md#python-vs-c).

Raw data: [`cpp.json`](cpp.json). Rerun with `make cpp && make bench EXP=cpp`. In the tables, `fastapi+uvicorn` is Python and `cpp` is C++.

## Throughput (requests/s)

| model | stack | workers | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 1 | 4,732 | 10,205 | 13,842 | 14,917 |
| noop | fastapi+uvicorn | 4 | 4,766 | 9,866 | 14,001 | 14,417 |
| noop | cpp | 1 | 14,085 | 37,395 | 68,582 | 79,667 |
| noop | cpp | 4 | 14,504 | 37,060 | 67,602 | 78,292 |
| catboost | fastapi+uvicorn | 1 | 2,704 | 6,628 | 10,238 | 12,836 |
| catboost | fastapi+uvicorn | 4 | 2,753 | 5,366 | 8,830 | 12,055 |
| catboost | cpp | 1 | 12,843 | 35,090 | 62,352 | 73,387 |
| catboost | cpp | 4 | 12,540 | 34,848 | 63,757 | 78,920 |
| lstm | fastapi+uvicorn | 1 | 1,171 | 1,559 | 3,116 | 4,747 |
| lstm | fastapi+uvicorn | 4 | 1,129 | 3,758 | 4,740 | 8,028 |
| lstm | cpp | 1 | 1,479 | 1,648 | 3,218 | 5,497 |
| lstm | cpp | 4 | 1,476 | 4,362 | 5,353 | 10,449 |

## P99 latency (ms)

| model | stack | workers | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 1 | 0.24 | 0.48 | 1.42 | 7.42 |
| noop | fastapi+uvicorn | 4 | 0.23 | 0.46 | 1.37 | 5.26 |
| noop | cpp | 1 | 0.13 | 0.17 | 0.60 | 2.29 |
| noop | cpp | 4 | 0.08 | 0.17 | 0.60 | 2.31 |
| catboost | fastapi+uvicorn | 1 | 0.39 | 0.64 | 3.72 | 6.74 |
| catboost | fastapi+uvicorn | 4 | 0.39 | 1.04 | 2.37 | 10.49 |
| catboost | cpp | 1 | 0.09 | 0.17 | 0.58 | 3.03 |
| catboost | cpp | 4 | 0.09 | 0.18 | 0.59 | 2.29 |
| lstm | fastapi+uvicorn | 1 | 0.88 | 2.70 | 5.84 | 21.86 |
| lstm | fastapi+uvicorn | 4 | 1.95 | 1.23 | 4.48 | 14.19 |
| lstm | cpp | 1 | 0.70 | 2.48 | 5.11 | 12.02 |
| lstm | cpp | 4 | 0.70 | 2.11 | 8.53 | 12.45 |

**Client-bound levels.** At 16 and 64 users, the C++ `noop` and `catboost` rows are **client-bound**: the Python load generator's processes are at 91–98% CPU (marked `!` in the full table). So those numbers are lower bounds on what the server can do. With 6 client processes the same server reached 87k RPS on `noop`, and with 8 it dropped again, because client and server had run out of cores. The Python rows and the LSTM rows used at most 30% client CPU.

## Observations

- **Python costs ~0.15 ms per request before any model runs.** With `noop` at 1 user, Python does 4.7k RPS (P50 0.21 ms) and C++ 14.1k (P50 0.07 ms). Most of C++'s 0.07 ms is the Python client itself: the server reports ~0.01 ms of its own time per request.
- **Cheap model: C++ is 4.7× faster at 1 user and at least 5–6× under load.** CatBoost does 12.8k vs 2.7k RPS at 1 user, and at least 73–79k vs 12.8k at 64 users.
  - About a third of the 1-user gap (~0.08 of 0.29 ms) is CatBoost's own Python wrapper: ~80 µs per call, against **3 µs** for the same model through the C API. The wrapper costs more than the trees do ([breakdown](models.md#why-catboost-is-slow-from-python)).
  - Python's batching hides that cost under load (30 rows per call at 64 users). The C++ server barely needs to batch (2–11 rows).
- **Expensive model: C++ is 13–30% faster.** The LSTM runs the same kernels, and `predict()` takes the same ~5.0 ms per ~16-row batch in both servers at 64 users. C++ wins only on per-request overhead and queueing:
  - 1,476 vs 1,129 RPS at 1 user (P50 0.68 vs 0.85 ms).
  - 10,449 vs 8,028 at 64 users, with a P99 of 12.5 vs 14.2 ms.
- **One C++ weak spot: the LSTM tail at 4–16 users.** With 4 workers the C++ P99 is higher than Python's (2.1 vs 1.2 ms at 4 users, 8.5 vs 4.5 ms at 16), although its median is lower. Not investigated, and it's a single run. A worker scheduled on an efficiency core would explain it.
- **When C++ pays off: it depends on the per-request overhead relative to the model's cost.**
  - **Expensive model (milliseconds per call):** Python with batching and several workers gets within 25% of an optimized C++ server.
  - **Cheap model (microseconds per call):** Python's ~0.15 ms overhead per request dominates, and C++ is 5× faster or more. Within Python, the remedy is fewer, bigger requests ([experiment 2](codec.md)).

<details>
<summary>All metrics, with the server-side time breakdown</summary>

Fixed: strategy=batched, codec=json, threads=1, samples=1
Server-side ms are means per request; batch = rows per predict() call.

| model | stack | workers | users | RPS | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 1 | 1 | 4732 | 0.21 | 0.24 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 30% |
| noop | fastapi+uvicorn | 1 | 4 | 10205 | 0.38 | 0.48 | 0.02 | 0.19 | 0.00 | 0.00 | 2.5 | 0 | 18% |
| noop | fastapi+uvicorn | 1 | 16 | 13842 | 1.15 | 1.42 | 0.02 | 0.60 | 0.00 | 0.00 | 10.4 | 0 | 21% |
| noop | fastapi+uvicorn | 1 | 64 | 14917 | 4.16 | 7.42 | 0.02 | 2.56 | 0.00 | 0.00 | 42.9 | 0 | 20% |
| noop | fastapi+uvicorn | 4 | 1 | 4766 | 0.21 | 0.23 | 0.02 | 0.07 | 0.00 | 0.00 | 1.0 | 0 | 29% |
| noop | fastapi+uvicorn | 4 | 4 | 9866 | 0.40 | 0.46 | 0.02 | 0.19 | 0.00 | 0.00 | 1.6 | 0 | 17% |
| noop | fastapi+uvicorn | 4 | 16 | 14001 | 1.13 | 1.37 | 0.02 | 0.56 | 0.00 | 0.00 | 8.9 | 0 | 21% |
| noop | fastapi+uvicorn | 4 | 64 | 14417 | 4.42 | 5.26 | 0.02 | 3.12 | 0.00 | 0.00 | 20.1 | 0 | 20% |
| noop | cpp | 1 | 1 | 14085 | 0.07 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 67% |
| noop | cpp | 1 | 4 | 37395 | 0.10 | 0.17 | 0.00 | 0.01 | 0.00 | 0.00 | 1.2 | 0 | 66% |
| noop | cpp | 1 | 16 | 68582 | 0.20 | 0.60 | 0.01 | 0.01 | 0.00 | 0.00 | 2.3 | 0 | 95% ! |
| noop | cpp | 1 | 64 | 79667 | 0.68 | 2.29 | 0.01 | 0.02 | 0.00 | 0.01 | 3.3 | 0 | 98% ! |
| noop | cpp | 4 | 1 | 14504 | 0.07 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 66% |
| noop | cpp | 4 | 4 | 37060 | 0.10 | 0.17 | 0.00 | 0.01 | 0.00 | 0.00 | 1.2 | 0 | 66% |
| noop | cpp | 4 | 16 | 67602 | 0.20 | 0.60 | 0.01 | 0.01 | 0.00 | 0.00 | 2.1 | 0 | 94% ! |
| noop | cpp | 4 | 64 | 78292 | 0.70 | 2.31 | 0.01 | 0.01 | 0.00 | 0.00 | 2.3 | 0 | 98% ! |
| catboost | fastapi+uvicorn | 1 | 1 | 2704 | 0.37 | 0.39 | 0.02 | 0.07 | 0.10 | 0.01 | 1.0 | 0 | 19% |
| catboost | fastapi+uvicorn | 1 | 4 | 6628 | 0.60 | 0.64 | 0.02 | 0.20 | 0.13 | 0.01 | 2.5 | 0 | 12% |
| catboost | fastapi+uvicorn | 1 | 16 | 10238 | 1.44 | 3.72 | 0.02 | 0.78 | 0.28 | 0.01 | 8.9 | 0 | 17% |
| catboost | fastapi+uvicorn | 1 | 64 | 12836 | 4.78 | 6.74 | 0.02 | 2.71 | 1.00 | 0.01 | 30.7 | 0 | 19% |
| catboost | fastapi+uvicorn | 4 | 1 | 2753 | 0.36 | 0.39 | 0.02 | 0.07 | 0.10 | 0.01 | 1.0 | 0 | 19% |
| catboost | fastapi+uvicorn | 4 | 4 | 5366 | 0.72 | 1.04 | 0.02 | 0.28 | 0.18 | 0.01 | 1.4 | 0 | 11% |
| catboost | fastapi+uvicorn | 4 | 16 | 8830 | 1.81 | 2.37 | 0.02 | 0.97 | 0.30 | 0.01 | 4.7 | 0 | 15% |
| catboost | fastapi+uvicorn | 4 | 64 | 12055 | 5.20 | 10.49 | 0.02 | 2.77 | 0.93 | 0.01 | 29.7 | 0 | 18% |
| catboost | cpp | 1 | 1 | 12843 | 0.08 | 0.09 | 0.00 | 0.00 | 0.01 | 0.00 | 1.0 | 0 | 66% |
| catboost | cpp | 1 | 4 | 35090 | 0.11 | 0.17 | 0.00 | 0.01 | 0.01 | 0.00 | 1.2 | 0 | 63% |
| catboost | cpp | 1 | 16 | 62352 | 0.23 | 0.58 | 0.01 | 0.02 | 0.02 | 0.01 | 3.0 | 0 | 91% ! |
| catboost | cpp | 1 | 64 | 73387 | 0.71 | 3.03 | 0.01 | 0.07 | 0.06 | 0.02 | 11.0 | 0 | 93% ! |
| catboost | cpp | 4 | 1 | 12540 | 0.08 | 0.09 | 0.00 | 0.00 | 0.01 | 0.00 | 1.0 | 0 | 67% |
| catboost | cpp | 4 | 4 | 34848 | 0.11 | 0.18 | 0.00 | 0.01 | 0.01 | 0.00 | 1.2 | 0 | 63% |
| catboost | cpp | 4 | 16 | 63757 | 0.22 | 0.59 | 0.01 | 0.01 | 0.02 | 0.00 | 1.9 | 0 | 94% ! |
| catboost | cpp | 4 | 64 | 78920 | 0.70 | 2.29 | 0.01 | 0.01 | 0.02 | 0.01 | 2.4 | 0 | 98% ! |
| lstm | fastapi+uvicorn | 1 | 1 | 1171 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+uvicorn | 1 | 4 | 1559 | 2.56 | 2.70 | 0.03 | 0.75 | 1.57 | 0.01 | 2.5 | 0 | 3% |
| lstm | fastapi+uvicorn | 1 | 16 | 3116 | 5.04 | 5.84 | 0.02 | 2.17 | 2.58 | 0.01 | 8.5 | 0 | 6% |
| lstm | fastapi+uvicorn | 1 | 64 | 4747 | 13.37 | 21.86 | 0.02 | 4.91 | 7.14 | 0.01 | 35.8 | 0 | 7% |
| lstm | fastapi+uvicorn | 4 | 1 | 1129 | 0.85 | 1.95 | 0.02 | 0.07 | 0.60 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+uvicorn | 4 | 4 | 3758 | 1.06 | 1.23 | 0.03 | 0.11 | 0.74 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+uvicorn | 4 | 16 | 4740 | 3.51 | 4.48 | 0.03 | 0.49 | 2.48 | 0.01 | 3.6 | 0 | 15% |
| lstm | fastapi+uvicorn | 4 | 64 | 8028 | 7.75 | 14.19 | 0.03 | 1.78 | 4.98 | 0.01 | 16.0 | 0 | 23% |
| lstm | cpp | 1 | 1 | 1479 | 0.67 | 0.70 | 0.00 | 0.00 | 0.55 | 0.00 | 1.0 | 0 | 8% |
| lstm | cpp | 1 | 4 | 1648 | 2.42 | 2.48 | 0.00 | 0.74 | 1.54 | 0.00 | 2.5 | 0 | 3% |
| lstm | cpp | 1 | 16 | 3218 | 4.97 | 5.11 | 0.00 | 1.72 | 3.00 | 0.01 | 12.5 | 0 | 6% |
| lstm | cpp | 1 | 64 | 5497 | 11.63 | 12.02 | 0.01 | 3.74 | 7.34 | 0.04 | 43.3 | 0 | 9% |
| lstm | cpp | 4 | 1 | 1476 | 0.68 | 0.70 | 0.00 | 0.00 | 0.55 | 0.00 | 1.0 | 0 | 9% |
| lstm | cpp | 4 | 4 | 4362 | 0.86 | 2.11 | 0.00 | 0.00 | 0.80 | 0.00 | 1.1 | 0 | 8% |
| lstm | cpp | 4 | 16 | 5353 | 3.18 | 8.53 | 0.01 | 0.31 | 2.42 | 0.01 | 4.1 | 0 | 16% |
| lstm | cpp | 4 | 64 | 10449 | 5.92 | 12.45 | 0.01 | 0.71 | 4.96 | 0.02 | 17.0 | 0 | 27% |

</details>

# VPS Performance — Actual Results vs Prediction

**Date:** 2026-03-20
**VPS:** 204.168.146.65 — Ubuntu 6.8.0, 4 vCPU, 7.6 GB RAM (KVM/x86_64)
**Setup:** Docker container (`python:3.11-slim` + CPU-only PyTorch), model baked into image
**Client:** macOS Darwin 25.3.0 (Apple Silicon), `load_test.py --duration 30 --concurrency 10 --no-spawn`
**Measured RTT:** avg 14 ms, max 73 ms (vs 20 ms assumed in prediction)

---

## Payload Sizes (per request, batch of 100 vessels)

| Protocol | Request | Response | Round-trip |
| -------- | ------: | -------: | ---------: |
| HTTP JSON (FastAPI / Flask) | 220.7 KB | 57.3 KB | **278.0 KB** |
| gRPC Protobuf               |  79.5 KB | 18.0 KB |  **97.5 KB** |

**Protobuf is ~2.9× smaller than JSON** per round-trip.
At the observed throughputs, this translates directly into data transfer rates:

| Server            | req/s | Upload (req) | Download (resp) | Total bandwidth |
| ----------------- | ----: | -----------: | --------------: | --------------: |
| FastAPI (1 worker) | 22.5 |    4.8 MB/s |        1.3 MB/s |       6.1 MB/s |
| FastAPI (4 workers)| 12.7 |    2.7 MB/s |        0.7 MB/s |       3.4 MB/s |
| Flask              | 36.6 |    7.9 MB/s |        2.0 MB/s |       9.9 MB/s |
| gRPC               | 39.0 |    3.0 MB/s |        0.7 MB/s |       3.7 MB/s |

*gRPC matches Flask throughput (req/s) but uses only **37% of the bandwidth** — binary encoding pays off on both serialization CPU and network.*

---

## Actual Results — Single Worker (baseline)

| Server  |   req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms | Mem (idle) |
| ------- | ------: | --------: | ------: | ------: | ------: | ---------: |
| FastAPI (1 worker) |    22.5 |     2 250 |  449.21 |  695.77 |  814.26 |    ~437 MB |
| Flask   |    36.6 |     3 655 |  263.87 |  367.31 |  505.60 |    ~276 MB |
| gRPC    | **39.0** | **3 902** | **252.89** | **335.62** | **379.81** | **~240 MB** |

## FastAPI — 4 Workers (follow-up test)

The initial result identified FastAPI's single-worker Uvicorn default as a bottleneck.
A second run was done with `workers=4` in `uvicorn.run()`.

| Server  |   req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms | Mem (idle) |
| ------- | ------: | --------: | ------: | ------: | ------: | ---------: |
| FastAPI (1 worker) |    22.5 |     2 250 |  449 ms |  696 ms |  814 ms |    ~437 MB |
| FastAPI (4 workers)|    12.7 |     1 272 |  790 ms | 1093 ms | 1204 ms |  **~1.08 GB** |

**4 workers made things worse.** Throughput dropped 44% and p50 latency nearly doubled.

### Why 4 workers hurt performance

Uvicorn's `--workers` creates separate Python *processes*, not threads. Each worker:
1. **Loads the model independently** → 4 × ~270 MB = ~1.08 GB RSS (confirmed by `docker stats`)
2. **Gets its own asyncio thread pool** (default: `min(32, cpu_count+4)` = 8 threads per worker) → 4 × 8 = 32 threads competing for 4 vCPUs

With the single-worker setup, the thread pool already saturated all 4 vCPUs with 8 inference threads.
Adding 3 more workers multiplied thread count 4× on the same CPUs, massively increasing context-switching and cache pressure.

**Flask's 4 threads and gRPC's 4-worker threadpool work correctly** because they are deliberately sized to match the vCPU count (1 thread per vCPU, no over-subscription).

### The right FastAPI fix

Set `--workers 1` (default) but limit the thread pool to 4 to match vCPU count:
```python
# server_fastapi.py — configure thread pool size to match vCPUs
import asyncio, concurrent.futures
loop = asyncio.get_event_loop()
loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=4))
```
Or restructure inference as a true async endpoint using `run_in_executor` with an explicit bounded pool.

*Memory measured via `docker stats --no-stream` at idle after run; CPU not captured (`--no-spawn` mode).*

---

## Prediction vs Actual

| Server  | Predicted req/s | Actual req/s | Δ req/s | Predicted p50 | Actual p50 | Δ p50   |
| ------- | --------------: | -----------: | ------: | ------------: | ---------: | ------: |
| FastAPI |            ~100 |         22.5 |   **−77%** |        ~100 ms |   449 ms |  **+349%** |
| Flask   |            ~105 |         36.6 |   **−65%** |         ~95 ms |   264 ms |  **+178%** |
| gRPC    |            ~115 |         39.0 |   **−66%** |         ~90 ms |   253 ms |  **+181%** |

| Server  | Predicted Mem | Actual Mem | Accuracy |
| ------- | ------------: | ---------: | -------: |
| FastAPI |       ~490 MB |    ~437 MB |     +11% |
| Flask   |       ~320 MB |    ~276 MB |     +14% |
| gRPC    |       ~310 MB |    ~240 MB |     +22% |

---

## What the Prediction Got Right

### 1. Ranking order — exactly correct
gRPC > Flask > FastAPI for both throughput and latency, as predicted.

### 2. gRPC's advantage over Flask — roughly correct
Prediction said gRPC beats Flask by ~10–15%.
Actual: gRPC beats Flask by ~6.5% on req/s, ~4% on p50 — same direction, similar magnitude.

### 3. FastAPI's relative collapse — directionally correct but more extreme
Prediction said FastAPI would hit the CPU ceiling sooner.
Actual: FastAPI p50 is **70% higher** than Flask/gRPC (449 ms vs 253–264 ms), far worse than the modest ~5% gap predicted.
The single-worker Uvicorn default serialises all 10 concurrent requests through one thread, creating a severe queue. Flask's 4-thread Waitress and gRPC's threadpool parallelise inference across cores.

### 4. Memory — accurate
Predictions within 10–22% of actuals, all in the correct direction.
FastAPI still uses ~45–80% more RAM than gRPC, as predicted.

### 5. p95/p50 ratio widens under network — correct
| Server  | Localhost ratio | Predicted VPS | Actual VPS |
| ------- | --------------: | ------------: | ---------: |
| FastAPI |           1.33× |         ~1.6× |      1.55× |
| Flask   |           1.19× |         ~1.5× |      1.39× |
| gRPC    |           1.22× |         ~1.4× |      1.33× |

All ratios widened as predicted. gRPC maintains the tightest spread.

---

## What the Prediction Got Wrong

### 1. Absolute latency — severely underestimated (biggest miss)

The prediction assumed VPS compute ≈ `p50_localhost + RTT`:
```
FastAPI: 71 ms + 20 ms RTT = ~91 ms (predicted ~100 ms)
Actual:  449 ms
```

Root cause: **Docker CPU-only PyTorch on x86 KVM is ~3.5× slower than Apple Silicon for LSTM inference.**

- macOS runs MPS/native AVX-optimised PyTorch; VPS runs generic x86 AVX2 in a virtualised container
- On localhost: batch of 100 vessels ≈ 70 ms → each vessel takes ~0.7 ms of compute
- On VPS: batch of 100 vessels ≈ 235–420 ms → compute is the bottleneck, not network

The 14 ms RTT is essentially invisible against 250+ ms of inference time.

### 2. Throughput — drastically overestimated

Prediction assumed `throughput ≈ concurrency / p50`:
```
10 workers / 0.091 s = ~110 req/s (FastAPI)
```

In practice, FastAPI/Uvicorn defaults to **1 worker**. All 10 concurrent requests queue behind a single event loop thread that still calls synchronous PyTorch inference. Result: p50 climbs to 449 ms as the queue builds.

Flask (Waitress 4 threads) and gRPC (4-worker threadpool) use real threading, so they serve ~4 requests in parallel → p50 stays near `batch_inference_time ≈ 250 ms`.

### 3. RTT assumption — wrong direction but irrelevant

Predicted 20 ms RTT; actual avg is **14 ms** (better, not worse). However, since inference dominates at 250–450 ms, RTT barely moves the needle.

---

## Key Takeaways

| Insight | Detail |
| ------- | ------ |
| VPS x86 CPU is ~3–4× slower than Apple Silicon for PyTorch LSTM | Inference dominates; RTT is negligible |
| FastAPI single-worker thread pool already saturates all vCPUs | Adding more workers multiplies threads and increases contention |
| gRPC wins on all axes | Binary encoding + threadpool = best throughput, lowest latency, lowest memory, lowest bandwidth |
| JSON request payload is 2.9× larger than protobuf | 220 KB vs 80 KB per request; matters at scale even on high-bandwidth links |
| Memory predictions were accurate | Model footprint is hardware-independent |
| 4-worker FastAPI is counterproductive on 4-vCPU CPU-bound workload | 1 worker + bounded thread pool (4 threads) is the correct fix |

---

## Baseline Reference

| Server  | Localhost req/s | VPS req/s | VPS/Localhost |
| ------- | --------------: | --------: | ------------: |
| FastAPI (1w) |         138 |      22.5 |        **16%** |
| FastAPI (4w) |         138 |      12.7 |         **9%** |
| Flask   |             132 |      36.6 |        **28%** |
| gRPC    |             141 |      39.0 |        **28%** |

The VPS delivers only **9–28% of localhost throughput** — far below the predicted 65–82%. The gap is almost entirely explained by the CPU speed difference per inference call. The 4-worker FastAPI result shows that naive "add more workers" tuning backfires on CPU-bound workloads when thread count exceeds vCPU count.

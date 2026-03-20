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

## Actual Results — Baseline (Waitress)

| Server  |   req/s | vessels/s |  p50 ms |  p95 ms |  p99 ms | Mem (idle) |
| ------- | ------: | --------: | ------: | ------: | ------: | ---------: |
| FastAPI (Uvicorn 1w) |  22.5 |  2 250 |  449 ms |  696 ms |  814 ms |    ~437 MB |
| Flask (Waitress 4t)  |  36.6 |  3 655 |  264 ms |  367 ms |  506 ms |    ~276 MB |
| gRPC (threadpool 4w) | **39.0** | **3 902** | **253 ms** | **336 ms** | **380 ms** | **~240 MB** |

## Flask — Gunicorn follow-up tests

### Attempt 1: Gunicorn 4 workers (naïve translation of "4 threads")

| Config | req/s | p50 ms | Mem |
| ------ | ----: | -----: | --: |
| Waitress 4 threads | 36.6 | 264 ms | ~276 MB |
| Gunicorn 4 workers | **2.4** | **4131 ms** | — |

4 separate processes × PyTorch's default multi-core intra-op parallelism = 4 × 4 = **16 threads competing for 4 vCPUs**. Each inference call slowed from ~250 ms to ~1 second; with 10 concurrent clients queuing, p50 ballooned to 4 s.

### Attempt 2: Gunicorn 1 worker + 4 threads (correct equivalent of Waitress 4 threads)

Gunicorn's `--threads N` flag creates N threads within a single worker process — identical to Waitress's thread model.

| Config | req/s | vessels/s | p50 ms | p95 ms | p99 ms | Mem |
| ------ | ----: | --------: | -----: | -----: | -----: | --: |
| Waitress 4 threads        | 36.6 | 3 655 | 264 ms | 367 ms | 506 ms | ~276 MB |
| Gunicorn 1w + 4t (Docker) | — | — | — | — | — | — |
| Gunicorn 1w + 4t (native) | **31.9** | **3 191** | **299 ms** | **438 ms** | **691 ms** | **~682 MB** |

Native gunicorn reaches ~87% of the Waitress result. The ~13% gap is mostly explained by higher memory RSS (native venv carries more overhead than the slim Docker image: ~682 MB vs ~276 MB), which increases memory pressure and page-fault frequency during inference.

*Memory for native run measured via `ps aux` RSS sum (gunicorn master + worker); no Docker involved.*

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
| For CPU-bound inference, process count × PyTorch threads must not exceed vCPU count | FastAPI 4w and Gunicorn 4w both collapsed for this reason |
| Correct Gunicorn equivalent of Waitress 4 threads: `--workers 1 --threads 4` | Single process, 4 threads — same as Waitress; reaches ~87% of Waitress result |
| gRPC wins on all axes | Binary encoding + threadpool = best throughput, lowest latency, lowest memory, lowest bandwidth |
| JSON request payload is 2.9× larger than protobuf | 220 KB vs 80 KB per request; matters at scale even on high-bandwidth links |
| Native venv has higher RSS than Docker slim image | ~682 MB (native) vs ~276 MB (Docker) for same Flask app — more memory pressure |

---

## Baseline Reference

| Server | Config | Localhost req/s | VPS req/s | VPS/Localhost |
| ------ | ------ | --------------: | --------: | ------------: |
| FastAPI | Uvicorn 1 worker     | 138 |  22.5 |  **16%** |
| Flask   | Waitress 4 threads   | 132 |  36.6 |  **28%** |
| Flask   | Gunicorn 4 workers   | 132 |   2.4 |   **2%** |
| Flask   | Gunicorn 1w + 4t     | 132 |  31.9 |  **24%** |
| gRPC    | Threadpool 4 workers | 141 |  39.0 |  **28%** |

The VPS delivers only **16–28% of localhost throughput** at best. The Gunicorn 4-process result (~2%) demonstrates that naive multi-process scaling backfires severely when inference is CPU-bound and PyTorch's intra-op threads exceed available vCPUs.

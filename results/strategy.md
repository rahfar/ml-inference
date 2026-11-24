# 1. Strategy: where `predict()` runs

Every strategy × every model. FastAPI on uvicorn, JSON, 1 sample per request, 4 torch threads per call, 1 inference worker. `noop` returns zeros, so its rows are the ceiling of the HTTP stack itself. The strategies are described in the [main README](../README.md#methodology).

Raw data: [`strategy.json`](strategy.json). Rerun with `make bench EXP=strategy`.

## Throughput (requests/s)

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

## P99 latency (ms)

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

## Observations

- **Never block the event loop with an expensive model.** The LSTM on `inline_loop` is stuck at ~1,520 RPS whatever the load, and its P99 at 64 users (82.6 ms) is twice its median, because requests queue unfairly in front of a busy loop. For `noop` and CatBoost, `inline_loop` is fine at low load because `predict()` is short and there's no thread hop.
- **Every thread hop costs something.** At the `noop` ceiling, handing work to a thread pool costs about 25% (12.1k vs 16.0k RPS at 64 users).
- **The LSTM scales with threads; CatBoost doesn't.** torch releases the GIL, so `inline_threadpool` runs LSTM predictions in parallel: 3,540 RPS at 4 users vs 1,757 for `serial` (whose throughput is simply 1 / 0.57 ms). CatBoost on the thread pool stays at ~5.7k: concurrent calls contend, and each one slows from 0.11 to 0.65 ms.
- **Batching wins where calls are cheap.** CatBoost on `batched` reaches 12,896 RPS at 64 users, 2.3× any other strategy, with a P99 of 6.6 ms instead of 13–22 ms. Average batches are 9 rows at 16 users and 31 at 64.
- **For the LSTM, one batching thread only pays off at high load.** At 4 users the average batch of 2.5 rows saves nothing, because a batch of 4 costs almost 4 single calls. At 16 users, parallel threads beat one batching thread (3,948 vs 2,544 RPS). At 64 users they're even (4,371 vs 4,108 RPS, same P99). The follow-up below shows why: with several inference workers, batching wins.

<details>
<summary>All metrics, with the server-side time breakdown</summary>

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

## Follow-up: batching with several inference workers

Batching lost for the LSTM above because it had only one inference thread:
- **torch's own threads barely parallelize this LSTM.** One call with 4 torch threads keeps only 1–2 cores busy, and is no faster than with 1 thread up to batch 64.
- **So `batched` effectively ran on one core,** while the thread pool ran up to 40 predictions across every core.

Two more runs, with the same bench and settings: `batched` with 1 or 4 inference workers (`--workers`, sharing one queue) × 1 or 4 torch threads, and the thread pool with 1 or 4 torch threads, for CatBoost and the LSTM.

Raw data: [`batched_workers.json`](batched_workers.json), [`threadpool_ref.json`](threadpool_ref.json). Rerun with `make bench EXP=batched_workers` and `make bench EXP=threadpool_ref`.

### Throughput (requests/s)

`batched`:

| model | workers | threads | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| catboost | 1 | 1 | 2,881 | 6,904 | 10,685 | 12,908 |
| catboost | 1 | 4 | 2,733 | 6,125 | 10,159 | 12,650 |
| catboost | 4 | 1 | 2,710 | 5,343 | 8,916 | 12,299 |
| catboost | 4 | 4 | 2,685 | 5,076 | 8,386 | 12,144 |
| lstm | 1 | 1 | 1,173 | 1,516 | 3,085 | 4,499 |
| lstm | 1 | 4 | 1,174 | 1,546 | 2,480 | 4,307 |
| lstm | 4 | 1 | 1,170 | 3,762 | 4,728 | 7,960 |
| lstm | 4 | 4 | 1,165 | 3,664 | 4,544 | 6,719 |

`inline_threadpool`:

| model | threads | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| catboost | 1 | 2,707 | 4,776 | 5,573 | 5,946 |
| catboost | 4 | 2,674 | 4,670 | 5,241 | 5,736 |
| lstm | 1 | 1,172 | 3,590 | 3,941 | 4,109 |
| lstm | 4 | 1,171 | 3,601 | 3,867 | 3,906 |

### P99 latency (ms)

`batched`:

| model | workers | threads | users 1 | users 4 | users 16 | users 64 |
|---|---|---|---:|---:|---:|---:|
| catboost | 1 | 1 | 0.38 | 0.63 | 2.10 | 6.71 |
| catboost | 1 | 4 | 0.40 | 0.72 | 2.20 | 6.85 |
| catboost | 4 | 1 | 0.39 | 1.04 | 2.36 | 6.38 |
| catboost | 4 | 4 | 0.41 | 1.15 | 2.48 | 5.76 |
| lstm | 1 | 1 | 0.88 | 3.69 | 6.01 | 39.87 |
| lstm | 1 | 4 | 0.88 | 2.73 | 7.07 | 22.65 |
| lstm | 4 | 1 | 0.88 | 1.23 | 4.57 | 14.48 |
| lstm | 4 | 4 | 0.88 | 2.33 | 5.02 | 14.16 |

`inline_threadpool`:

| model | threads | users 1 | users 4 | users 16 | users 64 |
|---|---|---:|---:|---:|---:|
| catboost | 1 | 0.40 | 1.10 | 3.94 | 11.30 |
| catboost | 4 | 0.40 | 1.13 | 7.10 | 13.25 |
| lstm | 1 | 0.88 | 1.41 | 6.69 | 22.05 |
| lstm | 4 | 0.88 | 1.40 | 6.81 | 51.37 |

### Observations

- **The LSTM needs several workers.** 4 workers × 1 torch thread: 7,960 RPS at 64 users, against 4,307 for 1 worker × 4 threads and 4,109 for the thread pool. It matches or beats the thread pool at every load (3,762 vs 3,590 at 4 users, 4,728 vs 3,941 at 16).
- **At 64 users the tail drops too,** from 22 to 14.5 ms P99.
- **1 torch thread per worker is best.** 4 workers × 4 threads is slower (6,719 RPS at 64 users), and each batch's `predict()` takes 7.1 ms instead of 5.0, likely because 16 threads compete for 10 cores.
- **For CatBoost, one worker was already right.** More workers only split the queue into smaller batches (12.3k vs 12.9k RPS at 64 users).
- **The rule:** use enough inference workers to keep up with the requests. That's one for a cheap model, and about one per performance core for an expensive one. Keep 1 torch thread per worker.

<details>
<summary>All metrics: batched</summary>

Fixed: stack=fastapi+uvicorn, strategy=batched, codec=json, samples=1
Server-side ms are means per request; batch = rows per predict() call.

| model | workers | threads | users | RPS | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | 1 | 1 | 1 | 2881 | 0.35 | 0.38 | 0.02 | 0.07 | 0.09 | 0.01 | 1.0 | 0 | 19% |
| catboost | 1 | 1 | 4 | 6904 | 0.58 | 0.63 | 0.02 | 0.21 | 0.12 | 0.01 | 2.5 | 0 | 13% |
| catboost | 1 | 1 | 16 | 10685 | 1.43 | 2.10 | 0.02 | 0.75 | 0.26 | 0.01 | 8.9 | 0 | 18% |
| catboost | 1 | 1 | 64 | 12908 | 4.75 | 6.71 | 0.02 | 2.73 | 0.97 | 0.01 | 30.6 | 0 | 19% |
| catboost | 1 | 4 | 1 | 2733 | 0.37 | 0.40 | 0.02 | 0.07 | 0.10 | 0.01 | 1.0 | 0 | 19% |
| catboost | 1 | 4 | 4 | 6125 | 0.65 | 0.72 | 0.02 | 0.24 | 0.15 | 0.01 | 2.5 | 0 | 11% |
| catboost | 1 | 4 | 16 | 10159 | 1.50 | 2.20 | 0.02 | 0.84 | 0.28 | 0.01 | 8.8 | 0 | 17% |
| catboost | 1 | 4 | 64 | 12650 | 4.85 | 6.85 | 0.02 | 2.81 | 0.95 | 0.01 | 30.8 | 0 | 19% |
| catboost | 4 | 1 | 1 | 2710 | 0.37 | 0.39 | 0.02 | 0.07 | 0.10 | 0.01 | 1.0 | 0 | 19% |
| catboost | 4 | 1 | 4 | 5343 | 0.73 | 1.04 | 0.02 | 0.28 | 0.18 | 0.01 | 1.4 | 0 | 11% |
| catboost | 4 | 1 | 16 | 8916 | 1.79 | 2.36 | 0.02 | 0.95 | 0.30 | 0.01 | 4.9 | 0 | 15% |
| catboost | 4 | 1 | 64 | 12299 | 5.21 | 6.38 | 0.02 | 2.75 | 0.87 | 0.01 | 30.3 | 0 | 18% |
| catboost | 4 | 4 | 1 | 2685 | 0.37 | 0.41 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 19% |
| catboost | 4 | 4 | 4 | 5076 | 0.77 | 1.15 | 0.02 | 0.32 | 0.19 | 0.01 | 1.3 | 0 | 10% |
| catboost | 4 | 4 | 16 | 8386 | 1.90 | 2.48 | 0.02 | 1.07 | 0.29 | 0.01 | 4.6 | 0 | 15% |
| catboost | 4 | 4 | 64 | 12144 | 5.27 | 5.76 | 0.02 | 2.91 | 0.89 | 0.01 | 26.1 | 0 | 18% |
| lstm | 1 | 1 | 1 | 1173 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 9% |
| lstm | 1 | 1 | 4 | 1516 | 2.55 | 3.69 | 0.03 | 0.80 | 1.59 | 0.01 | 2.5 | 0 | 3% |
| lstm | 1 | 1 | 16 | 3085 | 5.10 | 6.01 | 0.03 | 1.56 | 3.07 | 0.01 | 12.5 | 0 | 6% |
| lstm | 1 | 1 | 64 | 4499 | 13.49 | 39.87 | 0.02 | 5.79 | 7.04 | 0.01 | 32.0 | 0 | 6% |
| lstm | 1 | 4 | 1 | 1174 | 0.85 | 0.88 | 0.02 | 0.07 | 0.57 | 0.01 | 1.0 | 0 | 9% |
| lstm | 1 | 4 | 4 | 1546 | 2.57 | 2.73 | 0.03 | 0.74 | 1.59 | 0.01 | 2.5 | 0 | 3% |
| lstm | 1 | 4 | 16 | 2480 | 6.39 | 7.07 | 0.02 | 2.48 | 3.55 | 0.01 | 9.1 | 0 | 5% |
| lstm | 1 | 4 | 64 | 4307 | 14.53 | 22.65 | 0.02 | 4.38 | 8.66 | 0.01 | 43.3 | 0 | 6% |
| lstm | 4 | 1 | 1 | 1170 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 9% |
| lstm | 4 | 1 | 4 | 3762 | 1.06 | 1.23 | 0.03 | 0.11 | 0.74 | 0.01 | 1.0 | 0 | 8% |
| lstm | 4 | 1 | 16 | 4728 | 3.51 | 4.57 | 0.03 | 0.50 | 2.48 | 0.01 | 3.6 | 0 | 15% |
| lstm | 4 | 1 | 64 | 7960 | 7.87 | 14.48 | 0.03 | 1.80 | 5.01 | 0.01 | 16.5 | 0 | 22% |
| lstm | 4 | 4 | 1 | 1165 | 0.86 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | 4 | 4 | 4 | 3664 | 1.06 | 2.33 | 0.03 | 0.12 | 0.76 | 0.01 | 1.0 | 0 | 8% |
| lstm | 4 | 4 | 16 | 4544 | 3.69 | 5.02 | 0.03 | 0.52 | 2.60 | 0.01 | 3.7 | 0 | 15% |
| lstm | 4 | 4 | 64 | 6719 | 10.05 | 14.16 | 0.02 | 1.56 | 7.10 | 0.01 | 14.1 | 0 | 18% |

</details>

<details>
<summary>All metrics: inline_threadpool</summary>

Fixed: stack=fastapi+uvicorn, strategy=inline_threadpool, codec=json, workers=1, samples=1
Server-side ms are means per request; batch = rows per predict() call.

| model | threads | users | RPS | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | 1 | 1 | 2707 | 0.37 | 0.40 | 0.02 | 0.07 | 0.10 | 0.01 | 1.0 | 0 | 19% |
| catboost | 1 | 4 | 4776 | 0.83 | 1.10 | 0.02 | 0.31 | 0.25 | 0.01 | 1.0 | 0 | 9% |
| catboost | 1 | 16 | 5573 | 2.79 | 3.94 | 0.02 | 1.61 | 0.51 | 0.01 | 1.0 | 0 | 10% |
| catboost | 1 | 64 | 5946 | 10.70 | 11.30 | 0.02 | 7.53 | 0.58 | 0.01 | 1.0 | 0 | 10% |
| catboost | 4 | 1 | 2674 | 0.37 | 0.40 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 18% |
| catboost | 4 | 4 | 4670 | 0.85 | 1.13 | 0.02 | 0.34 | 0.22 | 0.01 | 1.0 | 0 | 9% |
| catboost | 4 | 16 | 5241 | 2.88 | 7.10 | 0.02 | 1.73 | 0.57 | 0.01 | 1.0 | 0 | 9% |
| catboost | 4 | 64 | 5736 | 11.04 | 13.25 | 0.02 | 7.80 | 0.64 | 0.01 | 1.0 | 0 | 9% |
| lstm | 1 | 1 | 1172 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | 1 | 4 | 3590 | 1.10 | 1.41 | 0.03 | 0.14 | 0.76 | 0.01 | 1.0 | 0 | 8% |
| lstm | 1 | 16 | 3941 | 3.93 | 6.69 | 0.03 | 1.44 | 1.87 | 0.01 | 1.0 | 0 | 11% |
| lstm | 1 | 64 | 4109 | 15.37 | 22.05 | 0.02 | 9.76 | 2.75 | 0.01 | 1.0 | 0 | 10% |
| lstm | 4 | 1 | 1171 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | 4 | 4 | 3601 | 1.09 | 1.40 | 0.03 | 0.14 | 0.76 | 0.01 | 1.0 | 0 | 8% |
| lstm | 4 | 16 | 3867 | 4.01 | 6.81 | 0.03 | 1.47 | 1.90 | 0.01 | 1.0 | 0 | 11% |
| lstm | 4 | 64 | 3906 | 15.47 | 51.37 | 0.02 | 10.42 | 2.75 | 0.01 | 1.0 | 0 | 10% |

</details>

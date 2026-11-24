# 3. Web stack: framework + HTTP server

Every web stack × every model. Strategy fixed to `inline_threadpool`: in every stack, `predict()` runs in one of 40 request threads, the only strategy a sync WSGI app can express. JSON, 1 sample per request, 4 torch threads per call.

Raw data: [`stack.json`](stack.json). Rerun with `make bench EXP=stack`.

## Throughput (requests/s)

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

## P99 latency (ms)

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

## Observations

- **gunicorn's thread worker is the bottleneck, not WSGI.** Flask on gunicorn (`gthread`) is stuck at ~2,800 RPS even with `noop`, and halves the LSTM's throughput (~2,000 vs ~4,000 RPS). The same Flask app on granian matches the async stacks, and is the fastest of all at 1 user (6,725 RPS on `noop`, 1,253 on the LSTM): granian parses HTTP in Rust and hands each request straight to a Python thread, without an event loop in between. Its weak spot is the tail at 64 users (P99 74.5 ms on `noop`), where 40 Python threads fight over the GIL.
- **FastAPI costs 15–20% over bare Starlette at the ceiling** (`noop` at 64 users: 12.1k vs 14.8k on uvicorn, 12.4k vs 15.6k on granian), but only 5–10% with CatBoost and 2–4% with the LSTM. The request handler here is the same in both, so that's FastAPI's own routing and request wrapping.
- **uvicorn vs granian is close to even under load.** On ASGI, granian is slower at 1 user (3.4–4.3k vs 5.1–5.3k RPS on `noop`; 6–8% slower with CatBoost and the LSTM) and level or slightly faster at 64 users.
- **With a real model, the stack matters much less than the strategy.** Across the five non-gunicorn stacks, CatBoost lands at 5.7–6.3k RPS and the LSTM at 3.8–4.2k at 64 users. Switching CatBoost from this thread-pool strategy to `batched` ([experiment 1](strategy.md)) gave 12.9k. Batching needs an async server, so that's the real argument for async: it makes batching easy. Raw speed isn't.

<details>
<summary>All metrics, with the server-side time breakdown</summary>

Fixed: strategy=inline_threadpool, codec=json, threads=4, samples=1
Server-side ms are means per request; batch = rows per predict() call.

| model | stack | users | RPS | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noop | fastapi+uvicorn | 1 | 5074 | 0.19 | 0.23 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 30% |
| noop | fastapi+uvicorn | 4 | 9878 | 0.40 | 0.49 | 0.02 | 0.19 | 0.00 | 0.00 | 1.0 | 0 | 17% |
| noop | fastapi+uvicorn | 16 | 11371 | 1.36 | 2.06 | 0.02 | 0.83 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | fastapi+uvicorn | 64 | 12089 | 5.21 | 6.23 | 0.02 | 3.79 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | starlette+uvicorn | 1 | 5312 | 0.19 | 0.21 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 33% |
| noop | starlette+uvicorn | 4 | 11883 | 0.33 | 0.40 | 0.02 | 0.17 | 0.00 | 0.00 | 1.0 | 0 | 20% |
| noop | starlette+uvicorn | 16 | 13178 | 1.14 | 2.82 | 0.02 | 0.75 | 0.00 | 0.00 | 1.0 | 0 | 22% |
| noop | starlette+uvicorn | 64 | 14771 | 4.30 | 5.24 | 0.02 | 3.11 | 0.00 | 0.00 | 1.0 | 0 | 22% |
| noop | fastapi+granian | 1 | 3441 | 0.29 | 0.32 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 22% |
| noop | fastapi+granian | 4 | 8220 | 0.48 | 0.60 | 0.02 | 0.15 | 0.00 | 0.00 | 1.0 | 0 | 15% |
| noop | fastapi+granian | 16 | 10857 | 1.44 | 2.16 | 0.02 | 0.55 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | fastapi+granian | 64 | 12426 | 5.04 | 8.33 | 0.02 | 1.88 | 0.00 | 0.00 | 1.0 | 0 | 19% |
| noop | starlette+granian | 1 | 4252 | 0.23 | 0.25 | 0.03 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 27% |
| noop | starlette+granian | 4 | 9977 | 0.40 | 0.50 | 0.02 | 0.12 | 0.00 | 0.00 | 1.0 | 0 | 18% |
| noop | starlette+granian | 16 | 12838 | 1.18 | 3.08 | 0.02 | 0.45 | 0.00 | 0.00 | 1.0 | 0 | 22% |
| noop | starlette+granian | 64 | 15584 | 4.03 | 6.42 | 0.02 | 1.51 | 0.00 | 0.00 | 1.0 | 0 | 23% |
| noop | flask+gunicorn | 1 | 2857 | 0.35 | 0.38 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 17% |
| noop | flask+gunicorn | 4 | 2703 | 1.46 | 2.45 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 6% |
| noop | flask+gunicorn | 16 | 2774 | 5.68 | 9.61 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 6% |
| noop | flask+gunicorn | 64 | 2812 | 22.58 | 29.03 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 6% |
| noop | flask+granian | 1 | 6725 | 0.15 | 0.17 | 0.03 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 39% |
| noop | flask+granian | 4 | 12636 | 0.25 | 1.88 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 22% |
| noop | flask+granian | 16 | 12094 | 0.76 | 6.29 | 0.03 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 21% |
| noop | flask+granian | 64 | 15541 | 1.60 | 74.51 | 0.02 | 0.00 | 0.00 | 0.00 | 1.0 | 0 | 25% |
| catboost | fastapi+uvicorn | 1 | 2688 | 0.37 | 0.40 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 18% |
| catboost | fastapi+uvicorn | 4 | 4676 | 0.85 | 1.13 | 0.02 | 0.34 | 0.22 | 0.01 | 1.0 | 0 | 9% |
| catboost | fastapi+uvicorn | 16 | 5466 | 2.86 | 4.01 | 0.02 | 1.65 | 0.55 | 0.01 | 1.0 | 0 | 9% |
| catboost | fastapi+uvicorn | 64 | 5785 | 10.97 | 13.02 | 0.02 | 7.74 | 0.64 | 0.01 | 1.0 | 0 | 9% |
| catboost | starlette+uvicorn | 1 | 2820 | 0.36 | 0.38 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 19% |
| catboost | starlette+uvicorn | 4 | 5127 | 0.77 | 1.04 | 0.02 | 0.30 | 0.21 | 0.01 | 1.0 | 0 | 10% |
| catboost | starlette+uvicorn | 16 | 5714 | 2.65 | 6.47 | 0.02 | 1.59 | 0.53 | 0.01 | 1.0 | 0 | 10% |
| catboost | starlette+uvicorn | 64 | 6263 | 10.14 | 12.16 | 0.02 | 7.15 | 0.64 | 0.01 | 1.0 | 0 | 10% |
| catboost | fastapi+granian | 1 | 2467 | 0.40 | 0.44 | 0.03 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 17% |
| catboost | fastapi+granian | 4 | 4457 | 0.89 | 1.15 | 0.02 | 0.22 | 0.19 | 0.01 | 1.0 | 0 | 9% |
| catboost | fastapi+granian | 16 | 5266 | 2.97 | 4.43 | 0.02 | 1.05 | 0.54 | 0.01 | 1.0 | 0 | 10% |
| catboost | fastapi+granian | 64 | 5748 | 10.86 | 17.32 | 0.02 | 3.86 | 1.36 | 0.01 | 1.0 | 0 | 10% |
| catboost | starlette+granian | 1 | 2594 | 0.38 | 0.42 | 0.03 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 18% |
| catboost | starlette+granian | 4 | 4899 | 0.81 | 1.05 | 0.02 | 0.19 | 0.17 | 0.01 | 1.0 | 0 | 10% |
| catboost | starlette+granian | 16 | 5734 | 2.74 | 4.09 | 0.02 | 0.92 | 0.49 | 0.01 | 1.0 | 0 | 11% |
| catboost | starlette+granian | 64 | 6313 | 9.94 | 15.34 | 0.02 | 3.42 | 1.29 | 0.01 | 1.0 | 0 | 11% |
| catboost | flask+gunicorn | 1 | 2127 | 0.47 | 0.50 | 0.02 | 0.00 | 0.11 | 0.01 | 1.0 | 0 | 14% |
| catboost | flask+gunicorn | 4 | 2111 | 1.86 | 3.07 | 0.02 | 0.00 | 0.38 | 0.01 | 1.0 | 0 | 5% |
| catboost | flask+gunicorn | 16 | 2166 | 7.25 | 12.17 | 0.02 | 0.00 | 1.04 | 0.01 | 1.0 | 0 | 5% |
| catboost | flask+gunicorn | 64 | 2167 | 29.27 | 38.38 | 0.02 | 0.00 | 1.46 | 0.01 | 1.0 | 0 | 5% |
| catboost | flask+granian | 1 | 3253 | 0.31 | 0.34 | 0.03 | 0.00 | 0.11 | 0.01 | 1.0 | 0 | 22% |
| catboost | flask+granian | 4 | 5663 | 0.69 | 1.03 | 0.03 | 0.00 | 0.29 | 0.01 | 1.0 | 0 | 11% |
| catboost | flask+granian | 16 | 5629 | 2.81 | 5.58 | 0.03 | 0.00 | 1.27 | 0.01 | 1.0 | 0 | 11% |
| catboost | flask+granian | 64 | 5757 | 8.83 | 37.69 | 0.02 | 0.00 | 6.07 | 0.01 | 1.0 | 0 | 11% |
| lstm | fastapi+uvicorn | 1 | 1172 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+uvicorn | 4 | 3593 | 1.09 | 1.41 | 0.03 | 0.14 | 0.76 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+uvicorn | 16 | 3899 | 3.98 | 6.72 | 0.03 | 1.46 | 1.88 | 0.01 | 1.0 | 0 | 11% |
| lstm | fastapi+uvicorn | 64 | 4065 | 15.58 | 22.11 | 0.02 | 9.86 | 2.78 | 0.01 | 1.0 | 0 | 11% |
| lstm | starlette+uvicorn | 1 | 1195 | 0.84 | 0.87 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | starlette+uvicorn | 4 | 3752 | 1.05 | 1.34 | 0.03 | 0.12 | 0.75 | 0.01 | 1.0 | 0 | 8% |
| lstm | starlette+uvicorn | 16 | 4084 | 3.80 | 6.50 | 0.03 | 1.32 | 1.87 | 0.01 | 1.0 | 0 | 13% |
| lstm | starlette+uvicorn | 64 | 4240 | 14.36 | 39.15 | 0.02 | 9.38 | 2.70 | 0.01 | 1.0 | 0 | 11% |
| lstm | fastapi+granian | 1 | 1104 | 0.90 | 0.94 | 0.03 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+granian | 4 | 3492 | 1.13 | 1.40 | 0.03 | 0.11 | 0.72 | 0.01 | 1.0 | 0 | 8% |
| lstm | fastapi+granian | 16 | 3891 | 4.03 | 6.23 | 0.03 | 0.84 | 1.66 | 0.01 | 1.0 | 0 | 12% |
| lstm | fastapi+granian | 64 | 4038 | 15.47 | 25.33 | 0.03 | 3.43 | 3.92 | 0.01 | 1.0 | 0 | 11% |
| lstm | starlette+granian | 1 | 1124 | 0.89 | 0.92 | 0.03 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | starlette+granian | 4 | 3666 | 1.08 | 1.30 | 0.03 | 0.09 | 0.71 | 0.01 | 1.0 | 0 | 8% |
| lstm | starlette+granian | 16 | 4193 | 3.74 | 5.79 | 0.03 | 0.76 | 1.59 | 0.01 | 1.0 | 0 | 13% |
| lstm | starlette+granian | 64 | 4145 | 14.47 | 48.75 | 0.03 | 3.56 | 3.74 | 0.01 | 1.0 | 0 | 11% |
| lstm | flask+gunicorn | 1 | 1060 | 0.94 | 0.98 | 0.02 | 0.00 | 0.58 | 0.01 | 1.0 | 0 | 7% |
| lstm | flask+gunicorn | 4 | 2043 | 1.95 | 2.83 | 0.03 | 0.00 | 0.85 | 0.01 | 1.0 | 0 | 5% |
| lstm | flask+gunicorn | 16 | 1986 | 7.96 | 13.15 | 0.03 | 0.00 | 2.83 | 0.01 | 1.0 | 0 | 5% |
| lstm | flask+gunicorn | 64 | 1997 | 31.62 | 46.35 | 0.03 | 0.00 | 5.36 | 0.01 | 1.0 | 0 | 5% |
| lstm | flask+granian | 1 | 1253 | 0.79 | 0.83 | 0.03 | 0.00 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | flask+granian | 4 | 4047 | 0.97 | 1.19 | 0.03 | 0.00 | 0.75 | 0.01 | 1.0 | 0 | 9% |
| lstm | flask+granian | 16 | 3949 | 3.79 | 9.20 | 0.04 | 0.00 | 3.10 | 0.02 | 1.0 | 0 | 13% |
| lstm | flask+granian | 64 | 3773 | 15.44 | 39.88 | 0.04 | 0.00 | 10.21 | 0.02 | 1.0 | 0 | 12% |

</details>

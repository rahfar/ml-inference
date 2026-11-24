# 2. Codec: how the array travels

Every codec × every model, at 1, 32 and 512 samples per request and 1 or 16 users. FastAPI on uvicorn, strategy fixed to `batched` (the overall winner of [experiment 1](strategy.md)), 4 torch threads per call, 1 inference worker.

Raw data: [`codec.json`](codec.json). Rerun with `make bench EXP=codec`.

## Throughput (samples/s = requests/s × samples per request)

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

## Server-side decode + encode time per request (ms)

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

## P99 latency (ms)

| model | codec | samples 1, users 1 | samples 1, users 16 | samples 32, users 1 | samples 32, users 16 | samples 512, users 1 | samples 512, users 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| noop | json | 0.24 | 1.41 | 1.00 | 13.16 | 12.25 | 173.52 |
| noop | msgpack | 0.20 | 1.00 | 0.40 | 3.40 | 2.68 | 34.57 |
| noop | protobuf | 0.21 | 0.99 | 0.23 | 1.60 | 0.58 | 6.45 |
| noop | raw | 0.20 | 0.93 | 0.30 | 1.23 | 0.27 | 2.38 |
| catboost | json | 0.41 | 2.16 | 1.57 | 16.72 | 17.15 | 227.79 |
| catboost | msgpack | 0.37 | 1.57 | 0.59 | 4.32 | 4.18 | 48.29 |
| catboost | protobuf | 0.37 | 1.55 | 0.46 | 2.94 | 1.69 | 21.83 |
| catboost | raw | 0.37 | 1.48 | 0.40 | 2.67 | 1.41 | 18.20 |
| lstm | json | 0.88 | 6.78 | 7.24 | 280.01 | 61.41 | 822.80 |
| lstm | msgpack | 0.85 | 6.70 | 6.35 | 86.48 | 55.99 | 1158.08 |
| lstm | protobuf | 0.85 | 7.02 | 6.21 | 85.61 | 47.04 | 728.97 |
| lstm | raw | 0.84 | 17.38 | 6.26 | 103.86 | 46.47 | 722.82 |

## Observations

- **For single samples, the format hardly matters for the LSTM** (within 5%). For `noop` and CatBoost, JSON costs about 30% of throughput at 16 users (13.9k vs 21.4k, and 10.3k vs 14.7k samples/s). Server-side decoding and encoding take only 0.03 ms, so most of that cost lies elsewhere. JSON bodies are 5× larger, and the client parses JSON too.
- **For large payloads, JSON becomes the bottleneck.** At 512 samples per request, decoding and encoding take ~13 ms per request in JSON, against 1.9 ms in msgpack, 0.25 ms in protobuf and ~0 in raw. For CatBoost that's a 12× throughput difference between JSON and raw (37.6k vs 457k samples/s). For the LSTM the model still dominates (10.1k vs 11.5k).
- **msgpack sits in between.** It's binary, but still a list of Python floats packed one by one. Protobuf's packed repeated float is a single buffer, close to raw. So the serialization part of gRPC's speed is protobuf's packed arrays; its HTTP/2 transport wasn't measured here.
- **Sending many samples per request is the biggest lever for a cheap model.** CatBoost with raw goes from 14.7k samples/s at 1 per request to 457k at 512 per request (31×). The LSTM goes from 2.5k to 11.5k (4.6×), as the [model profile](models.md#model-only-cost) predicts.
- **P99 with large payloads is queueing, not the codec.** At 512 samples and 16 users, each request waits behind ~15 others, so the LSTM's median is already 710–810 ms. That's the throughput limit showing as latency. A few P99 cells are single-run outliers (e.g. the LSTM with JSON at 32 samples and 16 users).

<details>
<summary>All metrics, with the server-side time breakdown</summary>

Fixed: strategy=batched, threads=4
Server-side ms are means per request; batch = rows per predict() call.

| model | codec | samples | users | RPS | samples/s | P50 ms | P99 ms | decode ms | wait ms | predict ms | encode ms | batch | errors | client CPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noop | json | 1 | 1 | 4822 | 4822 | 0.20 | 0.24 | 0.02 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 31% |
| noop | json | 1 | 16 | 13909 | 13909 | 1.15 | 1.41 | 0.02 | 0.59 | 0.00 | 0.00 | 10.4 | 0 | 21% |
| noop | json | 32 | 1 | 1041 | 33312 | 0.96 | 1.00 | 0.61 | 0.07 | 0.00 | 0.04 | 32.0 | 0 | 12% |
| noop | json | 32 | 16 | 1363 | 43622 | 11.80 | 13.16 | 0.61 | 9.56 | 0.00 | 0.04 | 64.0 | 0 | 4% |
| noop | json | 512 | 1 | 86 | 44134 | 11.58 | 12.25 | 9.60 | 0.08 | 0.00 | 0.62 | 512.0 | 0 | 8% |
| noop | json | 512 | 16 | 94 | 48230 | 169.67 | 173.52 | 9.66 | 126.82 | 0.00 | 0.63 | 512.0 | 0 | 2% |
| noop | msgpack | 1 | 1 | 5490 | 5490 | 0.18 | 0.20 | 0.00 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 34% |
| noop | msgpack | 1 | 16 | 19698 | 19698 | 0.81 | 1.00 | 0.00 | 0.44 | 0.00 | 0.00 | 9.2 | 0 | 30% |
| noop | msgpack | 32 | 1 | 2704 | 86541 | 0.37 | 0.40 | 0.10 | 0.07 | 0.00 | 0.02 | 32.0 | 0 | 24% |
| noop | msgpack | 32 | 16 | 5298 | 169542 | 3.09 | 3.40 | 0.10 | 2.46 | 0.00 | 0.02 | 64.0 | 0 | 13% |
| noop | msgpack | 512 | 1 | 383 | 195942 | 2.61 | 2.68 | 1.58 | 0.08 | 0.00 | 0.27 | 512.0 | 0 | 19% |
| noop | msgpack | 512 | 16 | 490 | 250931 | 32.66 | 34.57 | 1.58 | 27.25 | 0.00 | 0.27 | 512.0 | 0 | 7% |
| noop | protobuf | 1 | 1 | 5445 | 5445 | 0.18 | 0.21 | 0.00 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 34% |
| noop | protobuf | 1 | 16 | 19990 | 19990 | 0.79 | 0.99 | 0.00 | 0.44 | 0.00 | 0.00 | 9.1 | 0 | 30% |
| noop | protobuf | 32 | 1 | 4843 | 154979 | 0.20 | 0.23 | 0.00 | 0.06 | 0.00 | 0.02 | 32.0 | 0 | 31% |
| noop | protobuf | 32 | 16 | 10639 | 340461 | 1.50 | 1.60 | 0.00 | 1.25 | 0.00 | 0.02 | 64.0 | 0 | 19% |
| noop | protobuf | 512 | 1 | 1817 | 930099 | 0.55 | 0.58 | 0.01 | 0.07 | 0.00 | 0.23 | 512.0 | 0 | 18% |
| noop | protobuf | 512 | 16 | 2558 | 1309696 | 6.25 | 6.45 | 0.02 | 5.20 | 0.00 | 0.24 | 512.0 | 0 | 8% |
| noop | raw | 1 | 1 | 5692 | 5692 | 0.17 | 0.20 | 0.00 | 0.06 | 0.00 | 0.00 | 1.0 | 0 | 34% |
| noop | raw | 1 | 16 | 21448 | 21448 | 0.74 | 0.93 | 0.00 | 0.41 | 0.00 | 0.00 | 9.1 | 0 | 30% |
| noop | raw | 32 | 1 | 5369 | 171821 | 0.18 | 0.30 | 0.00 | 0.07 | 0.00 | 0.00 | 32.0 | 0 | 33% |
| noop | raw | 32 | 16 | 14456 | 462605 | 1.11 | 1.23 | 0.00 | 0.93 | 0.00 | 0.00 | 64.0 | 0 | 22% |
| noop | raw | 512 | 1 | 4112 | 2105242 | 0.24 | 0.27 | 0.00 | 0.07 | 0.00 | 0.00 | 512.0 | 0 | 36% |
| noop | raw | 512 | 16 | 7037 | 3602790 | 2.27 | 2.38 | 0.00 | 2.02 | 0.00 | 0.00 | 512.0 | 0 | 18% |
| catboost | json | 1 | 1 | 2687 | 2687 | 0.37 | 0.41 | 0.02 | 0.07 | 0.11 | 0.01 | 1.0 | 0 | 19% |
| catboost | json | 1 | 16 | 10282 | 10282 | 1.48 | 2.16 | 0.02 | 0.83 | 0.27 | 0.01 | 8.8 | 0 | 17% |
| catboost | json | 32 | 1 | 729 | 23334 | 1.36 | 1.57 | 0.61 | 0.07 | 0.19 | 0.20 | 32.0 | 0 | 15% |
| catboost | json | 32 | 16 | 1012 | 32378 | 15.76 | 16.72 | 0.62 | 12.98 | 1.35 | 0.20 | 64.0 | 0 | 6% |
| catboost | json | 512 | 1 | 60 | 30515 | 16.72 | 17.15 | 9.72 | 0.09 | 1.13 | 3.20 | 512.0 | 0 | 14% |
| catboost | json | 512 | 16 | 73 | 37581 | 217.56 | 227.79 | 9.85 | 176.64 | 13.34 | 3.21 | 512.0 | 0 | 5% |
| catboost | msgpack | 1 | 1 | 2994 | 2994 | 0.33 | 0.37 | 0.00 | 0.07 | 0.10 | 0.00 | 1.0 | 0 | 19% |
| catboost | msgpack | 1 | 16 | 13866 | 13866 | 1.12 | 1.57 | 0.00 | 0.57 | 0.23 | 0.00 | 9.0 | 0 | 22% |
| catboost | msgpack | 32 | 1 | 1832 | 58621 | 0.54 | 0.59 | 0.10 | 0.07 | 0.18 | 0.02 | 32.0 | 0 | 16% |
| catboost | msgpack | 32 | 16 | 3845 | 123040 | 4.15 | 4.32 | 0.10 | 3.40 | 0.38 | 0.02 | 64.0 | 0 | 10% |
| catboost | msgpack | 512 | 1 | 271 | 138598 | 3.67 | 4.18 | 1.58 | 0.08 | 1.06 | 0.28 | 512.0 | 0 | 14% |
| catboost | msgpack | 512 | 16 | 379 | 194202 | 41.90 | 48.29 | 1.55 | 37.04 | 2.58 | 0.27 | 512.0 | 0 | 5% |
| catboost | protobuf | 1 | 1 | 2928 | 2928 | 0.34 | 0.37 | 0.00 | 0.07 | 0.11 | 0.00 | 1.0 | 0 | 19% |
| catboost | protobuf | 1 | 16 | 14086 | 14086 | 1.11 | 1.55 | 0.00 | 0.56 | 0.23 | 0.00 | 9.1 | 0 | 23% |
| catboost | protobuf | 32 | 1 | 2382 | 76221 | 0.42 | 0.46 | 0.00 | 0.07 | 0.18 | 0.02 | 32.0 | 0 | 16% |
| catboost | protobuf | 32 | 16 | 5697 | 182298 | 2.80 | 2.94 | 0.00 | 2.29 | 0.31 | 0.02 | 64.0 | 0 | 11% |
| catboost | protobuf | 512 | 1 | 614 | 314266 | 1.63 | 1.69 | 0.01 | 0.08 | 1.05 | 0.25 | 512.0 | 0 | 6% |
| catboost | protobuf | 512 | 16 | 743 | 380211 | 21.54 | 21.83 | 0.01 | 19.66 | 1.30 | 0.23 | 512.0 | 0 | 3% |
| catboost | raw | 1 | 1 | 3013 | 3013 | 0.33 | 0.37 | 0.00 | 0.07 | 0.11 | 0.00 | 1.0 | 0 | 19% |
| catboost | raw | 1 | 16 | 14669 | 14669 | 1.06 | 1.48 | 0.00 | 0.54 | 0.22 | 0.00 | 8.9 | 0 | 23% |
| catboost | raw | 32 | 1 | 2709 | 86675 | 0.37 | 0.40 | 0.00 | 0.07 | 0.17 | 0.00 | 32.0 | 0 | 17% |
| catboost | raw | 32 | 16 | 6337 | 202778 | 2.52 | 2.67 | 0.00 | 2.07 | 0.27 | 0.00 | 64.0 | 0 | 12% |
| catboost | raw | 512 | 1 | 742 | 379699 | 1.35 | 1.41 | 0.00 | 0.08 | 1.04 | 0.00 | 512.0 | 0 | 7% |
| catboost | raw | 512 | 16 | 892 | 456858 | 17.93 | 18.20 | 0.00 | 16.52 | 1.07 | 0.00 | 512.0 | 0 | 3% |
| lstm | json | 1 | 1 | 1174 | 1174 | 0.85 | 0.88 | 0.02 | 0.07 | 0.58 | 0.01 | 1.0 | 0 | 8% |
| lstm | json | 1 | 16 | 2546 | 2546 | 6.27 | 6.78 | 0.02 | 1.67 | 4.07 | 0.01 | 12.5 | 0 | 5% |
| lstm | json | 32 | 1 | 141 | 4525 | 7.06 | 7.24 | 0.61 | 0.08 | 5.83 | 0.20 | 32.0 | 0 | 3% |
| lstm | json | 32 | 16 | 177 | 5658 | 87.25 | 280.01 | 0.69 | 78.15 | 11.34 | 0.21 | 64.0 | 0 | 1% |
| lstm | json | 512 | 1 | 17 | 8499 | 60.12 | 61.41 | 9.65 | 0.13 | 44.48 | 3.18 | 512.0 | 0 | 4% |
| lstm | json | 512 | 16 | 20 | 10086 | 812.73 | 822.80 | 10.24 | 744.38 | 50.60 | 3.22 | 512.0 | 0 | 1% |
| lstm | msgpack | 1 | 1 | 1220 | 1220 | 0.82 | 0.85 | 0.00 | 0.07 | 0.58 | 0.00 | 1.0 | 0 | 8% |
| lstm | msgpack | 1 | 16 | 2549 | 2549 | 6.26 | 6.70 | 0.00 | 1.87 | 4.05 | 0.00 | 12.5 | 0 | 5% |
| lstm | msgpack | 32 | 1 | 160 | 5130 | 6.23 | 6.35 | 0.10 | 0.08 | 5.82 | 0.02 | 32.0 | 0 | 2% |
| lstm | msgpack | 32 | 16 | 192 | 6150 | 83.10 | 86.48 | 0.11 | 72.40 | 10.35 | 0.02 | 64.0 | 0 | 1% |
| lstm | msgpack | 512 | 1 | 21 | 10752 | 47.34 | 55.99 | 1.62 | 0.12 | 44.68 | 0.28 | 512.0 | 0 | 1% |
| lstm | msgpack | 512 | 16 | 21 | 10803 | 721.88 | 1158.08 | 1.91 | 709.17 | 47.46 | 0.29 | 512.0 | 0 | 0% |
| lstm | protobuf | 1 | 1 | 1229 | 1229 | 0.82 | 0.85 | 0.00 | 0.07 | 0.58 | 0.00 | 1.0 | 0 | 8% |
| lstm | protobuf | 1 | 16 | 2531 | 2531 | 6.27 | 7.02 | 0.00 | 1.88 | 4.08 | 0.00 | 12.5 | 0 | 5% |
| lstm | protobuf | 32 | 1 | 164 | 5254 | 6.08 | 6.21 | 0.00 | 0.07 | 5.80 | 0.02 | 32.0 | 0 | 1% |
| lstm | protobuf | 32 | 16 | 194 | 6195 | 82.43 | 85.61 | 0.01 | 72.07 | 10.28 | 0.02 | 64.0 | 0 | 1% |
| lstm | protobuf | 512 | 1 | 22 | 11315 | 45.14 | 47.04 | 0.02 | 0.12 | 44.40 | 0.25 | 512.0 | 0 | 0% |
| lstm | protobuf | 512 | 16 | 22 | 11418 | 717.91 | 728.97 | 0.03 | 673.02 | 44.90 | 0.24 | 512.0 | 0 | 0% |
| lstm | raw | 1 | 1 | 1232 | 1232 | 0.81 | 0.84 | 0.00 | 0.07 | 0.57 | 0.00 | 1.0 | 0 | 8% |
| lstm | raw | 1 | 16 | 2461 | 2461 | 6.21 | 17.38 | 0.00 | 1.96 | 4.16 | 0.00 | 12.5 | 0 | 4% |
| lstm | raw | 32 | 1 | 164 | 5254 | 6.08 | 6.26 | 0.00 | 0.07 | 5.84 | 0.00 | 32.0 | 0 | 1% |
| lstm | raw | 32 | 16 | 193 | 6176 | 82.24 | 103.86 | 0.00 | 72.36 | 10.31 | 0.00 | 64.0 | 0 | 1% |
| lstm | raw | 512 | 1 | 22 | 11469 | 44.42 | 46.47 | 0.00 | 0.12 | 44.06 | 0.00 | 512.0 | 0 | 0% |
| lstm | raw | 512 | 16 | 22 | 11469 | 710.03 | 722.82 | 0.00 | 667.16 | 44.51 | 0.00 | 512.0 | 0 | 0% |

</details>

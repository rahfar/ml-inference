# VPS Deployment Performance Prediction

**Baseline:** macOS Darwin 25.3.0, Python 3.14, 20 s / 10 workers, 100 vessels/request (localhost).

---

## Baseline Results (localhost)

| Server  |   req/s |  vessels/s |    p50 ms |    p95 ms |    Mem avg | CPU avg |
| ------- | ------: | ---------: | --------: | --------: | ---------: | ------: |
| FastAPI |     138 |     13 780 |     71.45 |     94.87 |     490 MB |    659% |
| Flask   |     132 |     13 151 |     74.89 |     89.40 |     323 MB |    444% |
| gRPC    | **141** | **14 055** | **69.92** | **85.32** | **313 MB** |    458% |

---

## Predicted Results: Server on VPS, Load Test from Laptop

Assumptions: **20 ms RTT** (reasonable for a regional VPS), 4 vCPUs at ~60% of Apple Silicon per-core speed, 10 concurrent workers (unchanged).

| Server  | req/s (predicted) | vessels/s | p50 ms | p95 ms | Mem avg | CPU avg |
| ------- | ----------------: | --------: | -----: | -----: | ------: | ------: |
| FastAPI |              ~100 |   ~10 000 |  ~100  |  ~160  | ~490 MB |  ~400%+ |
| Flask   |              ~105 |   ~10 500 |   ~95  |  ~145  | ~320 MB |   ~300% |
| gRPC    |          **~115** | **~11 500** | **~90** | **~125** | **~310 MB** | **~280%** |

---

## Factor-by-Factor Analysis

### 1. Network RTT adds a floor to every request

On localhost the ~70 ms p50 is pure model compute — loopback latency is ~0.1 ms.
With a VPS the full network round trip is added to every request:

| RTT scenario | FastAPI p50 | Flask p50 | gRPC p50 |
| ------------ | ----------: | --------: | -------: |
| Localhost    |      71 ms  |    75 ms  |   70 ms  |
| +20 ms RTT   |     ~91 ms  |   ~95 ms  |  ~90 ms  |
| +40 ms RTT   |    ~111 ms  |  ~115 ms  | ~110 ms  |

Throughput at 10 workers ≈ `workers / p50`:

| RTT scenario | FastAPI req/s | Flask req/s | gRPC req/s |
| ------------ | ------------: | ----------: | ---------: |
| Localhost    |           138 |         132 |        141 |
| +20 ms RTT   |          ~110 |        ~105 |       ~111 |
| +40 ms RTT   |           ~90 |         ~87 |        ~91 |

**Throughput drops 20–35%** at typical RTT with 10 workers.
Raise `--concurrency` to 20–40 to partially recover it.

### 2. gRPC's wire-format advantage becomes tangible

Locally, JSON vs protobuf stays in memory — the format barely matters.
Over the network the response must travel as bytes:

- **JSON** (FastAPI / Flask): 100 vessels × 15 points × 2 coords × ~8 chars ≈ **~24 KB / response**
- **Protobuf** (gRPC): binary encoding is 5–10× smaller ≈ **~3–5 KB / response**

At 100 req/s JSON requires ~2.4 MB/s of download bandwidth on the laptop.
While this is within typical broadband capacity, the smaller payload means:

- Less serialization work on the client side
- Lower tail latency — p95/p50 ratio stays tighter under load
- gRPC's throughput lead over HTTP widens from **~2% → ~10–15%**

### 3. VPS CPU is slower per-core than Apple Silicon

The baseline used 659% CPU for FastAPI (≈ 6.6 Apple Silicon cores).
A 4-vCPU VPS core runs at roughly 50–60% of that speed per core.

- **FastAPI** will hit the 4-vCPU ceiling first — additional throughput loss of ~15–25% on top of the RTT penalty.
- **gRPC / Flask** (444–458% CPU locally) have more headroom and are less affected.
- FastAPI's relative disadvantage widens under VPS CPU constraints.

### 4. Memory profile stays roughly the same

Model weights (~3 MB on disk, ~100 MB in RAM) and Python runtime dominate RSS.
Expect similar absolute numbers (±10%). FastAPI will still consume ~50% more memory than gRPC/Flask due to JSON allocation — this is independent of deployment topology.

### 5. Latency variance (p95/p99 spread) increases

Loopback has near-zero jitter. Real network paths add variable queuing delay,
especially when the laptop load tester itself is under high async concurrency.

| Metric      | Localhost | VPS + laptop |
| ----------- | --------: | -----------: |
| p95/p50 ratio (FastAPI) | 1.33× | ~1.6×  |
| p95/p50 ratio (Flask)   | 1.19× | ~1.5×  |
| p95/p50 ratio (gRPC)    | 1.22× | ~1.4×  |

gRPC benefits most here: smaller messages traverse the network faster and with less jitter.

---

## Summary

| Factor                       | Localhost         | VPS + laptop load test            |
| ---------------------------- | ----------------- | --------------------------------- |
| p50 latency                  | 70–75 ms          | +15–40 ms (network RTT)           |
| Throughput (10 workers)      | 131–141 req/s     | ~90–115 req/s (↓ 20–35%)          |
| gRPC vs HTTP gap             | ~2%               | ~10–15% (wire format pays off)    |
| FastAPI overhead             | +50% mem, +45% CPU| Widens — CPU ceiling hit sooner   |
| p95/p50 ratio                | 1.2–1.3×          | 1.4–1.6×                          |
| Throughput recoverable?      | —                 | Yes — raise `--concurrency` to 30–40 |

**Bottom line:** gRPC becomes the clear winner in the VPS scenario.
Its binary encoding, lower bandwidth, and reduced client-side parsing work compound on top of its already-best compute efficiency.
FastAPI's JSON overhead, manageable on localhost, becomes a multi-layered liability: more bytes over the wire, more CPU to serialize/deserialize, and an earlier CPU ceiling on a constrained VPS.
Flask sits in the middle — competitive throughput with gRPC but still carrying HTTP/JSON costs.

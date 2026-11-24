# Results

The exact numbers behind the summary in the [main README](../README.md): one run per experiment, on the setup described there.

| File | Contents | Raw data |
|---|---|---|
| [models.md](models.md) | Model quality, model-only cost by batch size, why CatBoost is slow from Python | – |
| [strategy.md](strategy.md) | 1. Where `predict()` runs, and the follow-up with several inference workers | [strategy.json](strategy.json), [batched_workers.json](batched_workers.json), [threadpool_ref.json](threadpool_ref.json) |
| [codec.md](codec.md) | 2. Wire format at 1, 32 and 512 samples per request | [codec.json](codec.json) |
| [stack.md](stack.md) | 3. Web framework + HTTP server | [stack.json](stack.json) |
| [cpp.md](cpp.md) | 4. Python vs C++ | [cpp.json](cpp.json) |

## Columns

- **RPS:** requests per second. **samples/s:** RPS × samples per request.
- **P50, P99:** latency measured by the client, in ms.
- **decode, wait, predict, encode:** server-side time per request in ms, means of the `Server-Timing` header. `wait` runs from the end of decoding to the start of `predict()` (the queue or thread pool); time before the handler runs, such as behind a blocked event loop, isn't in it.
- **batch:** mean rows per `predict()` call.
- **client CPU:** the busiest load-generator process, in % of one core. `!` marks 80% or more: the client, not the server, may be the limit.

## Regenerating

The JSON files are what `bench.run` writes: one row per variant and concurrency level. A rerun (`make bench EXP=<name>`) overwrites its file; the tables in the `.md` files come from it:

```bash
uv run python -m bench.report results/strategy.json                 # every column
uv run python -m bench.report results/strategy.json --pivot p99     # one metric, a column per user count
uv run python -m bench.report results/codec.json --pivot samples/s --columns samples concurrency
```

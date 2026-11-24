# C++ server

The same API, models and timing headers as the Python server (`serving/`), built for speed. It shares no code with it: only the model weights in `models/weights/`.

| Part | Choice | Why |
|---|---|---|
| HTTP | [Drogon](https://github.com/drogonframework/drogon) | Async, one event loop per I/O thread, among the fastest C++ HTTP frameworks |
| LSTM | libtorch, TorchScript export (`export_lstm.py`) | The same ATen kernels as Python. AOTInductor was tried and was slower from batch 4 up. |
| CatBoost | `libcatboostmodel` (C API) | CatBoost's own evaluator, without the Python wrapper's ~80 µs per call |
| JSON | simdjson to parse, `std::to_chars` to write | |
| Build | `clang++ -O3 -mcpu=native -flto` | |

How requests flow (`--workers N`):
1. The Drogon event loops (`--io-threads`, default 2) parse HTTP and decode the body.
2. They push the request to a queue.
3. Each of N inference threads takes everything waiting, up to 64 rows, as one `predict()` call. This is the same batching as Python's `batched`, without the GIL.
4. That thread encodes each reply and hands it back to the request's event loop.

`--workers 0` runs `predict()` on the event loop instead.

## Build and run

```bash
brew install drogon simdjson     # once
make cpp                         # from the repo root: downloads libcatboostmodel, exports the LSTM, compiles

cpp/build/server --model lstm --workers 4 --threads 1       # serves on 127.0.0.1:8000
cpp/build/server --model catboost --profile                 # model-only cost, like `make profile`
```

Options: `--model lstm|catboost|noop`, `--codec json|raw`, `--threads N` (torch threads per call), `--workers N`, `--io-threads N`, `--host`, `--port`, `--weights DIR`.

The bench runs it for variants with `stack: cpp` (see the `cpp` experiment in `experiments.yaml`).

## Notes

- **libtorch comes from the project's venv.** `uv sync` installs it, and the binary links it through an rpath. The C++ and Python servers therefore use the same torch version.
- **CatBoost's C API has no thread setting.** Each `predict()` call is single-threaded, so parallelism comes from `--workers`.
- **JSON floats are shorter.** They're written as the shortest text that reads back as the same float32, while Python writes the float64 repr. The values are identical after decoding, and the bodies are smaller.

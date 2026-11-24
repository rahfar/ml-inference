# Models

Both models are trained on the same synthetic vessel tracks (`make train`): a 5-minute history (30 steps × 5 features = 150 floats) in, a 15-minute forecast (15 × 2 = 30 floats) out.

## Quality

Validation MSE in degrees². The baseline predicts that the vessel stays where it is.

| Model | Size | Val MSE |
|---|---|---:|
| last-position baseline | – | 5.147 |
| `lstm`: 2-layer LSTM, hidden 256 | 803k parameters, 3.2 MB | 0.0302 |
| `catboost`: MultiRMSE, 300 trees, depth 6 | 4.8 MB | 0.0155 |

## Model-only cost

`predict()` with no HTTP involved, by batch size (`make profile MODEL=lstm`, `make profile MODEL=catboost`; 4 threads, the setting of experiments 1–3):

| Batch | `lstm` ms/call | `lstm` µs/sample | `catboost` ms/call | `catboost` µs/sample |
|---:|---:|---:|---:|---:|
| 1 | 0.571 | 571 | 0.090 | 89.5 |
| 4 | 2.014 | 504 | 0.104 | 25.9 |
| 16 | 4.518 | 282 | 0.132 | 8.3 |
| 64 | 10.201 | 159 | 0.220 | 3.4 |
| 256 | 29.305 | 115 | 0.572 | 2.2 |

This table explains most of the results. The LSTM's cost per sample falls 5× from batch 1 to batch 256, and a batch of 4 costs almost as much as 4 single calls. CatBoost's per-sample cost falls **40×**: a call is mostly fixed overhead.

## Python vs C++

The same models, 1 thread, ms per call (`uv run python -m models.profile --model <m> --threads 1` vs `cpp/build/server --model <m> --profile`):

| Batch | `lstm` Python | `lstm` C++ | `catboost` Python | `catboost` C++ |
|---:|---:|---:|---:|---:|
| 1 | 0.573 | 0.554 | 0.084 | **0.003** |
| 4 | 2.040 | 2.022 | 0.082 | 0.011 |
| 16 | 3.569 | 3.530 | 0.109 | 0.027 |
| 64 | 9.946 | 9.886 | 0.194 | 0.088 |
| 256 | 36.697 | 36.527 | 0.557 | 0.334 |

- **The LSTM costs the same in both:** C++ runs the same ATen kernels through a TorchScript export. AOTInductor, PyTorch's ahead-of-time compiler, was also tried: 10% faster at batch 1, but 3–17% slower from batch 4 up.
- **CatBoost is 28× cheaper per single call in C++.** See below.

## Why CatBoost is slow from Python

A 1-row `CatBoostModel.predict` (1 thread), timed step by step with a one-off script:

| Step | µs |
|---|---:|
| **`CatBoostModel.predict`, total** | **81** |
| Our numpy pre/post-processing (`to_relative`, `from_offsets`) | 3 |
| Building a CatBoost `Pool` from the numpy array | 48 |
| CatBoost's internal predict on that `Pool` (`_base_predict`) | 25 |
| Python checks and argument handling in `regressor.predict()` | ~5 |
| Same model through the C library (`libcatboostmodel`), called from Python with ctypes | **6** |

- **`regressor.predict()` turns every input into a `Pool` first.** A `Pool` is CatBoost's dataset object, built for training. Its cost grows with the number of features, not rows: 9 µs for 1 row × 10 features, 48 µs for 1 × 150, 292 µs for 1 × 1,000, and still 50 µs for 64 × 150.
- **The predict step behind it adds about 22 µs per call** on top of evaluating the trees.
- **It isn't Python itself:** the C library called through ctypes takes 6 µs, close to C++'s 3 µs.

The overhead stays roughly fixed per call, and each row costs about the same either way. That's why batching helps Python CatBoost so much:

| Batch | `regressor.predict` | C library via ctypes |
|---:|---:|---:|
| 1 | 79 µs | 6 µs |
| 4 | 78 µs | 13 µs |
| 16 | 106 µs | 31 µs |
| 64 | 179 µs | 88 µs |
| 256 | 490 µs | 326 µs |

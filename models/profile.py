"""Model-only cost: predict() time vs batch size, no HTTP involved.

uv run python -m models.profile --model lstm --threads 4
"""

import argparse
import time

import models
from models.data import make_dataset

BATCH_SIZES = (1, 4, 16, 64, 256)


def time_predict(model: models.Model, x, budget_s: float = 1.0) -> float:
    """Mean seconds per predict(x) call."""
    for _ in range(3):  # warmup
        model.predict(x)
    calls, start = 0, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < budget_s:
        model.predict(x)
        calls += 1
    return elapsed / calls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=models.MODELS, default="lstm")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    model = models.load(args.model, args.threads)
    x, _ = make_dataset(max(BATCH_SIZES))
    print(f"{args.model}, {args.threads} threads")
    print(f"{'batch':>6} {'ms/call':>10} {'us/sample':>10}")
    for n in BATCH_SIZES:
        seconds = time_predict(model, x[:n])
        print(f"{n:>6} {seconds * 1e3:>10.3f} {seconds * 1e6 / n:>10.1f}")


if __name__ == "__main__":
    main()

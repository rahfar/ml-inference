"""Train the models on synthetic tracks; weights go to models/weights/.

    uv run python -m models.train                # all models
    uv run python -m models.train catboost       # just one

Both models learn offsets from the last observed position (see models/data.py).
Validation MSE is in degrees^2; "last position" is the trivial baseline that
predicts the vessel stops where it is.
"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import boosting, lstm
from models.data import (
    HISTORY_FEATURES,
    HISTORY_STEPS,
    make_dataset,
    to_offsets,
    to_relative,
)

N_TRAIN, N_VAL = 18_000, 2_000


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def train_lstm(x_train, t_train, x_val, t_val, epochs: int = 5) -> None:
    def as_seq(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).view(-1, HISTORY_STEPS, HISTORY_FEATURES)

    net = lstm.Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batches = DataLoader(
        TensorDataset(as_seq(x_train), torch.from_numpy(t_train)),
        batch_size=256,
        shuffle=True,
    )
    for epoch in range(1, epochs + 1):
        net.train()
        for xb, tb in batches:
            optimizer.zero_grad()
            loss_fn(net(xb), tb).backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            val = mse(net(as_seq(x_val)).numpy(), t_val)
        print(f"  lstm      epoch {epoch}/{epochs}  val MSE {val:.5f}")

    lstm.WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), lstm.WEIGHTS)
    params = sum(p.numel() for p in net.parameters())
    print(f"  lstm      {params:,} parameters -> {lstm.WEIGHTS}")


def train_catboost(x_train, t_train, x_val, t_val) -> None:
    regressor = boosting.new_regressor()
    regressor.fit(x_train, t_train, eval_set=(x_val, t_val))
    val = mse(regressor.predict(x_val), t_val)
    print(f"  catboost  val MSE {val:.5f}")

    boosting.WEIGHTS.parent.mkdir(exist_ok=True)
    regressor.save_model(str(boosting.WEIGHTS))
    print(
        f"  catboost  {boosting.TREES} trees, depth {boosting.DEPTH} -> {boosting.WEIGHTS}"
    )


TRAINERS = {"lstm": train_lstm, "catboost": train_catboost}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the models")
    p.add_argument("models", nargs="*", help=f"any of {list(TRAINERS)}; default: all")
    args = p.parse_args()
    if unknown := set(args.models) - set(TRAINERS):
        p.error(f"unknown model(s): {', '.join(unknown)}")

    x, y = make_dataset(N_TRAIN + N_VAL)
    inputs, targets = to_relative(x), to_offsets(x, y)
    split = (inputs[:N_TRAIN], targets[:N_TRAIN], inputs[N_TRAIN:], targets[N_TRAIN:])
    print(f"  baseline  last position  val MSE {mse(0, split[3]):.5f}")
    for name in args.models or TRAINERS:
        TRAINERS[name](*split)


if __name__ == "__main__":
    main()

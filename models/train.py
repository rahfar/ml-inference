"""Train the models on synthetic tracks; weights go to models/weights/.

uv run python -m models.train
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import lstm
from models.data import HISTORY_FEATURES, HISTORY_STEPS, make_dataset

N_TRAIN, N_VAL = 18_000, 2_000


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def train_lstm(x_train, y_train, x_val, y_val, epochs: int = 3) -> None:
    def as_seq(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).view(-1, HISTORY_STEPS, HISTORY_FEATURES)

    net = lstm.Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batches = DataLoader(
        TensorDataset(as_seq(x_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
    )
    for epoch in range(1, epochs + 1):
        net.train()
        for xb, yb in batches:
            optimizer.zero_grad()
            loss_fn(net(xb), yb).backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            val = mse(net(as_seq(x_val)).numpy(), y_val)
        print(f"  lstm  epoch {epoch}/{epochs}  val MSE {val:.4f}")

    lstm.WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), lstm.WEIGHTS)
    print(f"  saved {lstm.WEIGHTS}")


def main() -> None:
    x, y = make_dataset(N_TRAIN + N_VAL)
    x_train, y_train, x_val, y_val = x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]
    train_lstm(x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    main()

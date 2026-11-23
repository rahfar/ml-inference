"""PyTorch LSTM encoder -> linear decoder (~3 MB). Expensive: milliseconds per call."""

import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.data import HISTORY_FEATURES, HISTORY_STEPS, OUTPUT_SIZE

WEIGHTS = Path(__file__).parent / "weights" / "lstm.pt"


class Net(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(
            HISTORY_FEATURES, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.decoder = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n, 30, 5) -> (n, 30)"""
        _, (h, _) = self.encoder(x)
        return self.decoder(h[-1])


class LSTMModel:
    def __init__(self, threads: int):
        self.threads = threads
        self.net = Net()
        self.net.load_state_dict(torch.load(WEIGHTS, weights_only=True))
        self.net.eval()
        self._thread_state = threading.local()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """(n, 150) float32 -> (n, 30) float32"""
        # torch's thread count is per calling thread (OpenMP), so it has to be
        # set in whichever thread runs predict(): main, pool or executor thread.
        if getattr(self._thread_state, "threads", None) != self.threads:
            torch.set_num_threads(self.threads)
            self._thread_state.threads = self.threads
        with torch.inference_mode():
            y = self.net(torch.from_numpy(x).view(-1, HISTORY_STEPS, HISTORY_FEATURES))
        return y.numpy()

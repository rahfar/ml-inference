"""Models behind one interface: predict((n, 150) float32) -> (n, 30) float32."""

from typing import Protocol

import numpy as np

MODELS = ("lstm", "null")


class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def load(name: str, threads: int) -> Model:
    # Imported lazily so that e.g. the null model doesn't pull in torch.
    if name == "lstm":
        from models.lstm import LSTMModel

        return LSTMModel(threads)
    if name == "null":
        from models.null import NullModel

        return NullModel()
    raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")

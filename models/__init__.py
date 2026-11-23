"""Models behind one interface: predict((n, 150) float32) -> (n, 30) float32."""

from typing import Protocol

import numpy as np

MODELS = ("lstm", "catboost", "noop")


class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def load(name: str, threads: int) -> Model:
    # Imported lazily so that each server only loads its own library.
    if name == "lstm":
        from models.lstm import LSTMModel

        return LSTMModel(threads)
    if name == "catboost":
        from models.boosting import CatBoostModel

        return CatBoostModel(threads)
    if name == "noop":
        from models.noop import NoopModel

        return NoopModel()
    raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")

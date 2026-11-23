"""Does no work and returns zeros: measures the serving overhead on its own."""

import numpy as np

from models.data import OUTPUT_SIZE


class NoopModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        """(n, 150) float32 -> (n, 30) zeros"""
        return np.zeros((len(x), OUTPUT_SIZE), np.float32)

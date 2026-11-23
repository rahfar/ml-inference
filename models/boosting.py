"""CatBoost gradient-boosted trees: one multi-output model (MultiRMSE). Cheap per call."""

from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from models.data import from_offsets, to_relative

WEIGHTS = Path(__file__).parent / "weights" / "catboost.cbm"
TREES = 300
DEPTH = 6  # CatBoost's default


def new_regressor() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=TREES,
        depth=DEPTH,
        random_seed=0,
        verbose=100,
        allow_writing_files=False,  # no catboost_info/ training logs
    )


class CatBoostModel:
    def __init__(self, threads: int):
        self.threads = threads
        self.regressor = CatBoostRegressor()
        self.regressor.load_model(str(WEIGHTS))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """(n, 150) float32 -> (n, 30) float32"""
        offsets = self.regressor.predict(to_relative(x), thread_count=self.threads)
        return from_offsets(x, offsets.astype(np.float32))

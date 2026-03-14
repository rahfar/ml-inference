"""Train a CatBoost regression model targeting ~1 MB on disk."""
import os
import numpy as np
from catboost import CatBoostRegressor

os.makedirs("models", exist_ok=True)

np.random.seed(42)
N = 20_000
X = np.random.randn(N, 3).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + np.random.randn(N) * 0.1).astype(np.float32)

# depth=6, 800 trees → ~1 MB
model = CatBoostRegressor(
    iterations=800,
    depth=6,
    learning_rate=0.1,
    loss_function="RMSE",
    verbose=100,
    random_seed=42,
)
model.fit(X, y)

path = "models/catboost_model.cbm"
model.save_model(path)
size_kb = os.path.getsize(path) / 1024
print(f"\nSaved {path}  ({size_kb:.1f} KB)")

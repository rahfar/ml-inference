"""Train the vessel-track LSTM and save to models/pytorch_model.pt."""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model_def import (
    FUTURE_STEPS,
    HISTORY_FEATURES,
    HISTORY_STEPS,
    VesselTrackPredictor,
)

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic vessel-track dataset
# Each sample: initial state → simulate 30 history steps (10 s) +
#                               15 future steps (60 s)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N   = 20_000

lat = rng.uniform(54.0, 64.0, N).astype(np.float32)
lon = rng.uniform( 0.0, 20.0, N).astype(np.float32)
spd = rng.uniform( 5.0, 20.0, N).astype(np.float32)   # knots
crs = rng.uniform( 0.0, 2 * np.pi, N).astype(np.float32)


def advance(lat, lon, spd, crs, dt_hr):
    """Dead-reckoning step."""
    cos_lat = np.cos(np.radians(lat))
    return (
        lat + spd * dt_hr * np.cos(crs),
        lon + spd * dt_hr * np.sin(crs) / np.maximum(cos_lat, 0.01),
    )


history = np.empty((N, HISTORY_STEPS,  HISTORY_FEATURES), np.float32)
future  = np.empty((N, FUTURE_STEPS, 2),                  np.float32)

for t in range(HISTORY_STEPS):
    history[:, t, 0] = lat
    history[:, t, 1] = lon
    history[:, t, 2] = spd
    history[:, t, 3] = np.sin(crs)
    history[:, t, 4] = np.cos(crs)
    lat, lon = advance(lat, lon, spd, crs, 10 / 3600)
    spd += rng.normal(0, 0.05, N).astype(np.float32)
    crs += rng.normal(0, 0.005, N).astype(np.float32)

for t in range(FUTURE_STEPS):
    lat, lon = advance(lat, lon, spd, crs, 60 / 3600)
    spd += rng.normal(0, 0.1, N).astype(np.float32)
    crs += rng.normal(0, 0.02, N).astype(np.float32)
    future[:, t, 0] = lat
    future[:, t, 1] = lon

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
X = torch.from_numpy(history)
y = torch.from_numpy(future)

train_n = int(0.9 * N)
val_n   = N - train_n
train_ds, val_ds = random_split(TensorDataset(X, y), [train_n, val_n])
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256)

model     = VesselTrackPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training VesselTrackPredictor …")
for epoch in range(3):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(xb)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            val_loss += criterion(model(xb), yb).item() * len(xb)

    print(f"  epoch {epoch+1:2d}/20  train={tr_loss/train_n:.4f}  val={val_loss/val_n:.4f}")

path = "models/pytorch_model.pt"
torch.save(model.state_dict(), path)
size_kb = os.path.getsize(path) / 1024
print(f"\nSaved {path}  ({size_kb:.1f} KB)")

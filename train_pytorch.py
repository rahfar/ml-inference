"""Train a PyTorch MLP regression model targeting ~1 MB on disk."""
import os
import torch
import torch.nn as nn
import numpy as np
from model_def import MLP

os.makedirs("models", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N = 20_000
X_np = np.random.randn(N, 3).astype(np.float32)
y_np = (X_np[:, 0] * 2.0 + X_np[:, 1] - X_np[:, 2] + np.random.randn(N) * 0.1).astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

model.train()
for epoch in range(30):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30  loss={total_loss/len(loader):.4f}")

path = "models/pytorch_model.pt"
torch.save(model.state_dict(), path)
size_kb = os.path.getsize(path) / 1024
print(f"\nSaved {path}  ({size_kb:.1f} KB)")

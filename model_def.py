"""PyTorch MLP definition shared between training and serving."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """3 float inputs -> 1 float output, ~1 MB serialized."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

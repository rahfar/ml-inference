"""Vessel track predictor -- shared PyTorch model definition."""

import torch
import torch.nn as nn

HISTORY_STEPS = 30  # 5 min at 10-sec intervals
HISTORY_FEATURES = 5  # lat, lon, speed, course_sin, course_cos
FUTURE_STEPS = 15  # 15 min at 1-min intervals
FUTURE_FEATURES = 2  # lat, lon

INPUT_SIZE = HISTORY_STEPS * HISTORY_FEATURES  # 150
OUTPUT_SIZE = FUTURE_STEPS * FUTURE_FEATURES  # 30


class VesselTrackPredictor(nn.Module):
    """LSTM encoder -> linear decoder for vessel trajectory forecasting.

    Input : (batch, 30, 5)  -- 5-min history at 10-sec intervals
    Output: (batch, 15, 2)  -- 15-min forecast at 1-min intervals (lat, lon)
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(
            HISTORY_FEATURES,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.decoder = nn.Linear(hidden_size, FUTURE_STEPS * FUTURE_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        out = self.decoder(h[-1])
        return out.view(-1, FUTURE_STEPS, FUTURE_FEATURES)

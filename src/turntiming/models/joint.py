from __future__ import annotations

import torch
from torch import nn


class JointTimingModel(nn.Module):
    """Shared from-scratch audio encoder with turn-end and response-onset heads.

    The onset head predicts **seconds**, not milliseconds. Softplus keeps it
    non-negative, and working in seconds keeps the target on the same scale as
    the classification loss, so the two heads no longer need a magic divisor to
    stop the regression from swamping training.
    """

    def __init__(self, n_mels: int = 64, hidden_size: int = 96) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.turn_end_head = nn.Linear(hidden_size, 1)
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(features)
        state = encoded[:, -1]
        return {
            "turn_end_logit": self.turn_end_head(state).squeeze(-1),
            "onset_delay_s": self.onset_head(state).squeeze(-1),
        }

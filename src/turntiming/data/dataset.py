from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonlTimingDataset(Dataset):
    """Minimal dataset for synthetic or pre-extracted feature experiments.

    Expected JSONL fields:
      features: nested list [frames, mel]
      turn_end: 0/1
      time_to_next_speaker_ms: non-negative float
    """

    def __init__(self, path: str | Path) -> None:
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "turn_end": torch.tensor(row["turn_end"], dtype=torch.float32),
            "onset_ms": torch.tensor(row["time_to_next_speaker_ms"], dtype=torch.float32),
        }

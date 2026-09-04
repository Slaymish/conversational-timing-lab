"""Datasets for the two ways examples reach us.

``JsonlTimingDataset`` reads pre-extracted features inline -- fine for the
synthetic pipeline test, hopeless for real audio, where the features are
thousands of times larger than the labels.

``AudioWindowDataset`` reads a manifest of ``(audio_path, window_end_s, labels)``
and pulls the mel window out of the wav on demand, decoding only the frames it
needs. That is the format ``scripts/ingest.py`` produces.

Both yield the same batch keys, so training and evaluation code never has to
care which one it got. All times are seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from turntiming.data.schema import example_from_dict
from turntiming.features.audio import LogMelFrontend
from turntiming.features.io import read_window

BATCH_KEYS = ("features", "turn_end", "onset_s", "pause_s", "offset_s", "onset_valid")


def _label_tensors(row: dict) -> dict[str, torch.Tensor]:
    example = example_from_dict(row)
    return {
        "turn_end": torch.tensor(float(example.turn_end)),
        "onset_s": torch.tensor(float(example.time_to_next_speaker_s)),
        "pause_s": torch.tensor(float(example.pause_duration_s)),
        "offset_s": torch.tensor(float(example.observation_offset_s)),
        "onset_valid": torch.tensor(float(not example.onset_censored)),
    }


def _read_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class JsonlTimingDataset(Dataset):
    """Examples whose ``features`` are stored inline as [frames, mel]."""

    def __init__(self, path: str | Path) -> None:
        self.rows = _read_rows(path)
        if self.rows and "features" not in self.rows[0]:
            raise ValueError(f"{path} has no inline features; use AudioWindowDataset")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            **_label_tensors(row),
        }


class AudioWindowDataset(Dataset):
    """Log-mel windows decoded straight out of the source audio.

    Only the ``window_seconds`` ending at ``window_end_s`` is decoded, so a
    30-minute meeting costs no more than the windows actually sampled from it.
    Windows that would start before the file does are left-padded with silence.
    """

    def __init__(
        self,
        path: str | Path,
        window_seconds: float = 2.0,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        audio_root: str | Path | None = None,
    ) -> None:
        self.rows = _read_rows(path)
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.audio_root = Path(audio_root) if audio_root else None
        self.frontend = LogMelFrontend(sample_rate=sample_rate, n_mels=n_mels)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        missing = [r for r in self.rows if not r.get("audio_path")]
        if missing:
            raise ValueError(f"{len(missing)} rows in {path} have no audio_path")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, audio_path: str) -> Path:
        path = Path(audio_path)
        return self.audio_root / path.name if self.audio_root else path

    def _load_window(self, audio_path: str, window_end_s: float) -> torch.Tensor:
        block, rate = read_window(self._resolve(audio_path), window_end_s, self.window_seconds)
        waveform = torch.from_numpy(block)

        if rate != self.sample_rate:
            if rate not in self._resamplers:
                self._resamplers[rate] = torchaudio.transforms.Resample(rate, self.sample_rate)
            waveform = self._resamplers[rate](waveform)

        target = round(self.window_seconds * self.sample_rate)
        if waveform.numel() < target:
            waveform = torch.nn.functional.pad(waveform, (target - waveform.numel(), 0))
        return waveform[-target:]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        waveform = self._load_window(row["audio_path"], float(row["window_end_s"]))
        with torch.no_grad():
            features = self.frontend(waveform.unsqueeze(0)).squeeze(0)
        return {"features": features, **_label_tensors(row)}


def build_dataset(path: str | Path, **kwargs) -> Dataset:
    """Pick the dataset class that matches the manifest.

    Rows carrying inline ``features`` go to ``JsonlTimingDataset``; rows carrying
    an ``audio_path`` go to ``AudioWindowDataset``.
    """
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    if "features" in rows[0]:
        return JsonlTimingDataset(path)
    return AudioWindowDataset(path, **kwargs)

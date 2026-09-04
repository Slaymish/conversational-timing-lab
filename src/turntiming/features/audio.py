from __future__ import annotations

import torch
import torchaudio


class LogMelFrontend(torch.nn.Module):
    def __init__(self, sample_rate: int = 16_000, n_mels: int = 64) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert [batch, time] waveform to [batch, frames, mel]."""
        x = self.to_db(self.mel(waveform))
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / (
            x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        )
        return x.transpose(1, 2)

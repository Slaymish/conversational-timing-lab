"""Reading short windows out of long recordings.

A conversational corpus is a handful of very long files, and training touches
only a couple of seconds around each boundary. Decoding the whole meeting to
read 2 seconds of it makes ingestion the bottleneck, so windows are read at a
frame offset and nothing else is decoded.

``soundfile`` does the reading rather than ``torchaudio.load``, which in recent
versions delegates to TorchCodec and is unavailable without it. soundfile also
covers the formats speech corpora actually ship in -- wav, flac, ogg.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf


@lru_cache(maxsize=256)
def probe(path: str) -> tuple[int, int]:
    """Return ``(sample_rate, num_frames)`` without decoding audio."""
    info = sf.info(path)
    return info.samplerate, info.frames


def read_window(path: str | Path, window_end_s: float, window_seconds: float) -> tuple[np.ndarray, int]:
    """Read the ``window_seconds`` of mono audio ending at ``window_end_s``.

    Windows reaching past either end of the file are padded with silence, so a
    boundary near the start of a recording is still usable.
    """
    path = str(path)
    rate, total_frames = probe(path)

    want = round(window_seconds * rate)
    end_frame = min(round(window_end_s * rate), total_frames)
    start_frame = end_frame - want
    offset = max(0, start_frame)

    block, _ = sf.read(
        path, start=offset, frames=max(0, end_frame - offset), dtype="float32", always_2d=True
    )
    mono = block.mean(axis=1)

    if len(mono) < want:
        mono = np.pad(mono, (want - len(mono), 0))
    return mono[-want:], rate

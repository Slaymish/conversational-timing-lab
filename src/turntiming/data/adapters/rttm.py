"""NIST RTTM adapter, for anything you have already diarised.

Field order is the standard one::

    SPEAKER <file-id> <chan> <start> <duration> <ortho> <stype> <speaker> <conf> <slat>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from turntiming.data.schema import SpeakerSegment


def load_segments(path: str | Path) -> Iterator[SpeakerSegment]:
    """Yield segments from one RTTM file or a directory of them."""
    root = Path(path)
    files = sorted(root.glob("*.rttm")) if root.is_dir() else [root]
    for file in files:
        for line in file.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            start_s, duration_s = float(fields[3]), float(fields[4])
            if duration_s <= 0:
                continue
            yield SpeakerSegment(fields[1], fields[7], start_s, start_s + duration_s)

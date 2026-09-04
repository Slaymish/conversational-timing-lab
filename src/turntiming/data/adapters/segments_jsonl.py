"""Adapter for segments already in the canonical shape.

One JSON object per line::

    {"conversation_id": "c1", "speaker": "A", "start_s": 0.0, "end_s": 1.2, "text": "hi"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from turntiming.data.schema import SpeakerSegment


def load_segments(path: str | Path) -> Iterator[SpeakerSegment]:
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        yield SpeakerSegment(
            conversation_id=str(row["conversation_id"]),
            speaker=str(row["speaker"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            text=str(row.get("text", "")),
        )

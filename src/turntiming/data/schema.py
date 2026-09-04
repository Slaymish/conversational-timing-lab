"""Canonical records shared by every corpus adapter and every experiment.

Two record types matter:

``SpeakerSegment``
    A stretch of speech by one speaker, in seconds. This is the only thing a
    corpus adapter has to produce.

``TimingExample``
    One decision point for the model: "given everything up to ``window_end_s``,
    has this speaker yielded the floor, and when would a partner start?"

All times are stored in **seconds**. Millisecond values only appear in metric
reports, where they are easier to read.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpeakerSegment:
    """A contiguous stretch of speech attributed to one speaker."""

    conversation_id: str
    speaker: str
    start_s: float
    end_s: float
    text: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class TimingExample:
    """A single labelled decision point.

    ``window_end_s`` is the last moment the model is allowed to observe. It sits
    ``observation_offset_s`` after the speaker stopped, so ``pause_duration_s``
    is the silence heard *so far* -- never the silence that will eventually
    elapse. Keeping it causal is what stops the pause-duration baseline from
    quietly reading the future.
    """

    conversation_id: str
    speaker: str
    window_end_s: float
    ipu_end_s: float
    observation_offset_s: float
    pause_duration_s: float
    turn_end: int
    time_to_next_speaker_s: float
    next_speaker: str | None = None
    is_backchannel: int = 0
    is_overlap: int = 0
    onset_censored: int = 0
    audio_path: str | None = None
    text: str = ""
    features: list[list[float]] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if row["features"] is None:
            row.pop("features")
        return row


_EXAMPLE_FIELDS = {f.name for f in fields(TimingExample)}


def example_from_dict(row: dict[str, Any]) -> TimingExample:
    """Build an example from a JSONL row, tolerating legacy millisecond fields."""
    row = dict(row)
    if "time_to_next_speaker_s" not in row and "time_to_next_speaker_ms" in row:
        row["time_to_next_speaker_s"] = row.pop("time_to_next_speaker_ms") / 1000.0
    if "pause_duration_s" not in row and "pause_duration_ms" in row:
        row["pause_duration_s"] = row.pop("pause_duration_ms") / 1000.0
    return TimingExample(**{k: v for k, v in row.items() if k in _EXAMPLE_FIELDS})


def write_jsonl(path: str | Path, examples: Iterable[TimingExample]) -> int:
    """Write examples as JSONL, returning the number of rows written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict()) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[TimingExample]:
    """Read examples from JSONL, skipping blank lines."""
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield example_from_dict(json.loads(line))

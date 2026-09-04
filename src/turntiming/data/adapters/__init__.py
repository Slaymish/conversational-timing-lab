"""Corpus adapters.

An adapter's whole job is to turn some corpus's native format into
``SpeakerSegment`` records. Everything downstream -- IPU merging, labelling,
feature extraction, evaluation -- is corpus-agnostic, so adding a corpus means
adding one function here.
"""

from __future__ import annotations

from typing import Callable, Iterator

from turntiming.data.adapters import ami, rttm, segments_jsonl
from turntiming.data.schema import SpeakerSegment

ADAPTERS: dict[str, Callable[..., Iterator[SpeakerSegment]]] = {
    "ami": ami.load_segments,
    "rttm": rttm.load_segments,
    "jsonl": segments_jsonl.load_segments,
}


def load_segments(kind: str, path: str, **kwargs) -> Iterator[SpeakerSegment]:
    if kind not in ADAPTERS:
        raise KeyError(f"Unknown adapter {kind!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[kind](path, **kwargs)


__all__ = ["ADAPTERS", "load_segments", "ami", "rttm", "segments_jsonl"]

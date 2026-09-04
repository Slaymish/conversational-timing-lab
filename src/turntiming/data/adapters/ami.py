"""AMI Meeting Corpus adapter (CC BY 4.0).

Reads the NXT word-level annotations, which carry per-word start/end times for
each speaker on their own headset channel. Word timings are far tighter than
the transcriber segment boundaries, so we hand the words straight to the IPU
merger and let it decide where the pauses are.

Download with ``scripts/fetch_ami.py``; the annotations are a 23 MB zip.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path

from turntiming.data.schema import SpeakerSegment

_NITE = "{http://nite.sourceforge.net/}"
_WORDS_FILE = re.compile(r"^(?P<meeting>[^.]+)\.(?P<speaker>[^.]+)\.words\.xml$")


def available_meetings(annotations_dir: str | Path) -> list[str]:
    """List meeting ids that have word annotations."""
    words_dir = Path(annotations_dir) / "words"
    meetings = {
        match.group("meeting")
        for path in words_dir.glob("*.words.xml")
        if (match := _WORDS_FILE.match(path.name))
    }
    return sorted(meetings)


def load_segments(
    annotations_dir: str | Path,
    meetings: Sequence[str] | None = None,
    include_vocalsounds: bool = True,
) -> Iterator[SpeakerSegment]:
    """Yield one segment per annotated word (or vocal sound).

    Zero-length tokens -- punctuation and instantaneous markers -- are dropped,
    since they describe the transcript rather than the audio.
    """
    words_dir = Path(annotations_dir) / "words"
    if not words_dir.is_dir():
        raise FileNotFoundError(f"No AMI 'words' directory under {annotations_dir}")

    wanted = set(meetings) if meetings else None
    for path in sorted(words_dir.glob("*.words.xml")):
        match = _WORDS_FILE.match(path.name)
        if not match:
            continue
        meeting, speaker = match.group("meeting"), match.group("speaker")
        if wanted is not None and meeting not in wanted:
            continue

        for element in ET.parse(path).getroot():
            tag = element.tag.split("}")[-1]
            if tag == "w":
                if element.get("punc") == "true":
                    continue
                text = (element.text or "").strip()
            elif tag == "vocalsound" and include_vocalsounds:
                text = f"[{element.get('type', 'vocalsound')}]"
            else:
                continue

            start, end = element.get("starttime"), element.get("endtime")
            if start is None or end is None:
                continue
            start_s, end_s = float(start), float(end)
            if end_s <= start_s:
                continue
            yield SpeakerSegment(meeting, speaker, start_s, end_s, text)

"""Turn boundaries from speaker segments.

The one rule this module exists to enforce: **a silence is not a turn end**. A
boundary is labelled by *who speaks next*, not by how quiet it got. A speaker
who pauses for 800 ms and then carries on produces a negative example, and it
is the most valuable kind of negative we have.

Pipeline::

    segments -> inter-pausal units (IPUs) -> boundaries -> observed examples

A boundary is the end of an IPU. It is observed at several offsets after
speech stops, and an observation is only emitted while the outcome is still
unknown -- once somebody has started speaking, there is nothing left to predict.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from turntiming.data.schema import SpeakerSegment, TimingExample

DEFAULT_BACKCHANNELS = frozenset(
    {
        "yeah", "yep", "yes", "mm", "mmm", "mhm", "mm-hmm", "uh-huh", "huh",
        "right", "okay", "ok", "sure", "true", "exactly", "hmm", "oh", "ah",
        "wow", "really", "nice", "good", "cool", "great", "of course", "i see",
        "yeah yeah", "okay okay", "right right", "no", "absolutely",
    }
)

_PUNCT = re.compile(r"[^a-z' -]+")


@dataclass
class LabelConfig:
    """Knobs for turning segments into labelled decision points."""

    ipu_merge_gap_s: float = 0.18
    """Silence shorter than this stays *inside* one inter-pausal unit."""

    observation_offsets_s: Sequence[float] = (0.1, 0.3, 0.5, 1.0)
    """How long after speech stops the model is asked to decide."""

    min_ipu_duration_s: float = 0.20
    """Drop very short IPUs; they are usually noise or clipped annotations."""

    max_lookahead_s: float = 10.0
    """Onset targets are censored beyond this horizon."""

    backchannel_max_duration_s: float = 1.0
    backchannel_resume_window_s: float = 2.0
    backchannel_lexicon: frozenset[str] = field(default=DEFAULT_BACKCHANNELS)
    backchannel_requires_text: bool = True
    """If False, any short utterance the speaker talks straight through counts."""

    def __post_init__(self) -> None:
        self.observation_offsets_s = tuple(sorted(float(o) for o in self.observation_offsets_s))


def normalise_text(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def merge_ipus(
    segments: Iterable[SpeakerSegment], gap_s: float = 0.18
) -> list[SpeakerSegment]:
    """Collapse each speaker's segments into inter-pausal units.

    Segments from the same speaker separated by less than ``gap_s`` belong to
    one continuous run of speech. Speakers are merged independently, so
    overlapping talk survives.
    """
    by_speaker: dict[tuple[str, str], list[SpeakerSegment]] = defaultdict(list)
    for segment in segments:
        by_speaker[(segment.conversation_id, segment.speaker)].append(segment)

    ipus: list[SpeakerSegment] = []
    for (conversation_id, speaker), items in by_speaker.items():
        items.sort(key=lambda s: (s.start_s, s.end_s))
        start, end, words = items[0].start_s, items[0].end_s, [items[0].text]
        for segment in items[1:]:
            if segment.start_s - end <= gap_s:
                end = max(end, segment.end_s)
                words.append(segment.text)
            else:
                ipus.append(
                    SpeakerSegment(conversation_id, speaker, start, end, " ".join(w for w in words if w).strip())
                )
                start, end, words = segment.start_s, segment.end_s, [segment.text]
        ipus.append(
            SpeakerSegment(conversation_id, speaker, start, end, " ".join(w for w in words if w).strip())
        )

    ipus.sort(key=lambda s: (s.start_s, s.end_s))
    return ipus


def _is_backchannel(candidate: SpeakerSegment, cfg: LabelConfig) -> bool:
    if candidate.duration_s > cfg.backchannel_max_duration_s:
        return False
    text = normalise_text(candidate.text)
    if not text:
        return not cfg.backchannel_requires_text
    return text in cfg.backchannel_lexicon


def _boundary_outcome(
    ipu: SpeakerSegment, ipus: Sequence[SpeakerSegment], cfg: LabelConfig
) -> tuple[int, SpeakerSegment | None, int] | None:
    """Resolve one boundary into (turn_end, next_other_speaker_ipu, is_backchannel)."""
    end = ipu.end_s
    next_same: SpeakerSegment | None = None
    next_other: SpeakerSegment | None = None
    for other in ipus:
        if other is ipu or other.start_s <= end:
            continue
        if other.speaker == ipu.speaker:
            if next_same is None:
                next_same = other
        elif next_other is None:
            next_other = other
        if next_same is not None and next_other is not None:
            break

    if next_same is None and next_other is None:
        return None  # end of the conversation; nothing to predict

    if next_other is None:
        return 0, None, 0
    if next_same is None:
        return 1, next_other, 0

    if next_other.start_s < next_same.start_s:
        resumes_quickly = next_same.start_s - next_other.start_s <= cfg.backchannel_resume_window_s
        if resumes_quickly and _is_backchannel(next_other, cfg):
            # A "mm-hmm" over the top is not a floor transfer.
            return 0, next_other, 1
        return 1, next_other, 0
    return 0, next_other, 0


def build_examples(
    segments: Iterable[SpeakerSegment],
    cfg: LabelConfig | None = None,
    audio_paths: dict[str, str] | None = None,
    stats: Counter | None = None,
) -> Iterator[TimingExample]:
    """Turn raw speaker segments into labelled, causally observed decision points.

    Pass a ``Counter`` as ``stats`` to find out how many boundaries were seen and
    why some produced nothing -- worth watching, since overlapped boundaries are
    common in multi-party audio and are dropped on purpose.
    """
    cfg = cfg or LabelConfig()
    stats = stats if stats is not None else Counter()
    audio_paths = audio_paths or {}

    by_conversation: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in segments:
        if segment.end_s > segment.start_s:
            by_conversation[segment.conversation_id].append(segment)

    for conversation_id, raw in sorted(by_conversation.items()):
        ipus = merge_ipus(raw, cfg.ipu_merge_gap_s)
        conversation_end = max(ipu.end_s for ipu in ipus)
        audio_path = audio_paths.get(conversation_id)

        for ipu in ipus:
            stats["boundaries"] += 1
            if ipu.duration_s < cfg.min_ipu_duration_s:
                stats["skipped_short_ipu"] += 1
                continue

            # Somebody else talking straight through the boundary means there is
            # no pause to interpret: the floor was contested, not offered.
            if any(
                o is not ipu and o.speaker != ipu.speaker and o.start_s <= ipu.end_s < o.end_s
                for o in ipus
            ):
                stats["skipped_talked_over"] += 1
                continue

            outcome = _boundary_outcome(ipu, ipus, cfg)
            if outcome is None:
                stats["skipped_conversation_end"] += 1
                continue
            turn_end, next_other, is_backchannel = outcome

            # Informational: was this speaker overlapped at any point in the IPU?
            is_overlap = int(
                any(
                    o.speaker != ipu.speaker and o.start_s < ipu.end_s and o.end_s > ipu.start_s
                    for o in ipus
                )
            )

            if next_other is None:
                time_to_next, censored = cfg.max_lookahead_s, 1
            else:
                time_to_next = max(0.0, next_other.start_s - ipu.end_s)
                censored = int(time_to_next > cfg.max_lookahead_s)
                time_to_next = min(time_to_next, cfg.max_lookahead_s)

            # The boundary stops being a question the moment anybody speaks.
            silence_until = min(
                min((o.start_s for o in ipus if o.start_s > ipu.end_s), default=conversation_end),
                conversation_end,
            )

            emitted = 0
            for offset in cfg.observation_offsets_s:
                window_end = ipu.end_s + offset
                if window_end > silence_until:
                    break
                emitted += 1
                stats["examples"] += 1
                stats[f"examples_offset_{offset}"] += 1
                yield TimingExample(
                    conversation_id=conversation_id,
                    speaker=ipu.speaker,
                    window_end_s=round(window_end, 4),
                    ipu_end_s=round(ipu.end_s, 4),
                    observation_offset_s=offset,
                    pause_duration_s=offset,
                    turn_end=turn_end,
                    time_to_next_speaker_s=round(time_to_next, 4),
                    next_speaker=next_other.speaker if next_other else None,
                    is_backchannel=is_backchannel,
                    is_overlap=is_overlap,
                    onset_censored=censored,
                    audio_path=audio_path,
                    text=ipu.text,
                )
            if emitted == 0:
                stats["skipped_resolved_immediately"] += 1

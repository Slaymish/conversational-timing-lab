"""The rule under test: silence does not mean the turn ended."""

from collections import Counter

import pytest

from turntiming.data.label import LabelConfig, build_examples, merge_ipus
from turntiming.data.schema import SpeakerSegment


def seg(speaker: str, start: float, end: float, text: str = "word") -> SpeakerSegment:
    return SpeakerSegment("c1", speaker, start, end, text)


CFG = LabelConfig(observation_offsets_s=(0.1, 0.3))


def labels(segments, cfg=CFG) -> list[int]:
    return [e.turn_end for e in build_examples(segments, cfg)]


def test_same_speaker_resumes_is_not_a_turn_end() -> None:
    """'So what I was thinking was...' 400ms '...maybe we should' -- still their turn."""
    segments = [seg("A", 0.0, 2.0), seg("A", 2.4, 4.0), seg("B", 20.0, 21.0)]
    assert labels(segments)[:2] == [0, 0]


def test_other_speaker_takes_over_is_a_turn_end() -> None:
    """'Yeah, that sounds good.' 400ms, then B speaks -- floor transferred."""
    segments = [seg("A", 0.0, 2.0), seg("B", 2.4, 4.0)]
    assert labels(segments)[:2] == [1, 1]


def test_identical_silence_gets_opposite_labels() -> None:
    """The two cases above differ only in who speaks next, never in the pause."""
    hold = [seg("A", 0.0, 2.0), seg("A", 2.4, 4.0), seg("B", 20.0, 21.0)]
    shift = [seg("A", 0.0, 2.0), seg("B", 2.4, 4.0)]
    hold_example = next(iter(build_examples(hold, CFG)))
    shift_example = next(iter(build_examples(shift, CFG)))
    assert hold_example.pause_duration_s == shift_example.pause_duration_s
    assert hold_example.turn_end != shift_example.turn_end


def test_backchannel_does_not_take_the_floor() -> None:
    segments = [seg("A", 0.0, 2.0), seg("B", 2.4, 2.7, "mm-hmm"), seg("A", 3.0, 5.0)]
    examples = list(build_examples(segments, CFG))
    assert [e.turn_end for e in examples[:2]] == [0, 0]
    assert all(e.is_backchannel for e in examples[:2])


def test_long_interjection_does_take_the_floor() -> None:
    segments = [seg("A", 0.0, 2.0), seg("B", 2.4, 6.0, "actually I disagree"), seg("A", 7.0, 9.0)]
    assert next(iter(build_examples(segments, CFG))).turn_end == 1


def test_observations_stop_once_someone_speaks() -> None:
    """No example may be emitted from a window that already contains the answer."""
    segments = [seg("A", 0.0, 2.0), seg("B", 2.2, 4.0)]
    cfg = LabelConfig(observation_offsets_s=(0.1, 0.3, 0.5, 1.0))
    offsets = [e.observation_offset_s for e in build_examples(segments, cfg)]
    assert offsets == [0.1]  # B starts 0.2s in, so 0.3s onwards is unobservable


def test_boundary_talked_over_is_dropped() -> None:
    """A stops while B is mid-word: there is no pause here to interpret."""
    segments = [seg("A", 0.0, 2.0), seg("B", 1.5, 3.0), seg("A", 4.0, 5.0)]
    stats: Counter = Counter()
    examples = list(build_examples(segments, CFG, stats=stats))

    assert stats["skipped_talked_over"] == 1
    assert [e.speaker for e in examples] == ["B", "B"]  # only B's clean boundary survives
    assert all(e.is_overlap for e in examples)


def test_onset_target_is_seconds_and_censored_beyond_horizon() -> None:
    segments = [seg("A", 0.0, 2.0), seg("B", 2.5, 3.0)]
    example = next(iter(build_examples(segments, CFG)))
    assert example.time_to_next_speaker_s == pytest.approx(0.5)
    assert example.onset_censored == 0

    far = [seg("A", 0.0, 2.0), seg("A", 2.5, 3.0), seg("B", 90.0, 91.0)]
    censored = next(iter(build_examples(far, LabelConfig(observation_offsets_s=(0.1,), max_lookahead_s=10.0))))
    assert censored.time_to_next_speaker_s == 10.0
    assert censored.onset_censored == 1


def test_merge_ipus_joins_short_gaps_only() -> None:
    segments = [seg("A", 0.0, 1.0), seg("A", 1.1, 2.0), seg("A", 3.0, 4.0)]
    merged = merge_ipus(segments, gap_s=0.18)
    assert [(round(m.start_s, 2), round(m.end_s, 2)) for m in merged] == [(0.0, 2.0), (3.0, 4.0)]


def test_merge_ipus_keeps_speakers_separate() -> None:
    merged = merge_ipus([seg("A", 0.0, 1.0), seg("B", 1.05, 2.0)], gap_s=0.18)
    assert len(merged) == 2

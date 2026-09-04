import json

from turntiming.data.schema import TimingExample, example_from_dict, read_jsonl, write_jsonl


def make(**overrides) -> TimingExample:
    fields = {
        "conversation_id": "c1", "speaker": "A", "window_end_s": 2.1, "ipu_end_s": 2.0,
        "observation_offset_s": 0.1, "pause_duration_s": 0.1, "turn_end": 1,
        "time_to_next_speaker_s": 0.42,
    }
    return TimingExample(**{**fields, **overrides})


def test_legacy_millisecond_fields_are_converted() -> None:
    example = example_from_dict(
        {
            "conversation_id": "c1", "speaker": "A", "window_end_s": 2.1, "ipu_end_s": 2.0,
            "observation_offset_s": 0.1, "pause_duration_ms": 100, "turn_end": 1,
            "time_to_next_speaker_ms": 420,
        }
    )
    assert example.time_to_next_speaker_s == 0.42
    assert example.pause_duration_s == 0.1


def test_unknown_fields_are_ignored() -> None:
    assert example_from_dict({**make().to_dict(), "some_future_field": 1}).turn_end == 1


def test_roundtrip_jsonl(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    assert write_jsonl(path, [make(), make(turn_end=0)]) == 2
    assert [e.turn_end for e in read_jsonl(path)] == [1, 0]


def test_features_are_omitted_when_absent(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [make()])
    assert "features" not in json.loads(path.read_text().splitlines()[0])

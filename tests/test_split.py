import numpy as np
import pytest

from turntiming.data.split import group_keys, group_split


def rows(n_conversations: int, per_conversation: int) -> list[dict]:
    return [
        {"conversation_id": f"c{c}", "speaker": "A", "ipu_end_s": float(i)}
        for c in range(n_conversations)
        for i in range(per_conversation)
    ]


def test_conversations_never_straddle_the_split() -> None:
    data = rows(10, 8)
    keys = group_keys(data, "conversation")
    train_idx, val_idx = group_split(keys, val_fraction=0.3, seed=0)

    train_ids = {data[i]["conversation_id"] for i in train_idx}
    val_ids = {data[i]["conversation_id"] for i in val_idx}
    assert train_ids.isdisjoint(val_ids)
    assert len(train_idx) + len(val_idx) == len(data)


def test_boundary_split_keeps_offsets_of_one_boundary_together() -> None:
    data = [
        {"conversation_id": "c1", "speaker": "A", "ipu_end_s": b, "observation_offset_s": o}
        for b in range(40)
        for o in (0.1, 0.3, 0.5, 1.0)
    ]
    keys = group_keys(data, "boundary")
    train_idx, val_idx = group_split(keys, val_fraction=0.25, seed=0)

    train_boundaries = {data[i]["ipu_end_s"] for i in train_idx}
    val_boundaries = {data[i]["ipu_end_s"] for i in val_idx}
    assert train_boundaries.isdisjoint(val_boundaries)


def test_split_is_deterministic() -> None:
    keys = group_keys(rows(12, 5), "conversation")
    first = group_split(keys, 0.2, seed=7)
    second = group_split(keys, 0.2, seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_single_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="split_by"):
        group_split(group_keys(rows(1, 20), "conversation"), 0.2)


def test_unknown_split_key() -> None:
    with pytest.raises(ValueError, match="split_by must be"):
        group_keys(rows(2, 2), "nonsense")

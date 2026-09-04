"""Splits that do not leak.

Each boundary emits several rows -- one per observation offset -- and those rows
share a speaker, a room, and almost the same audio. Split rows at random and
near-duplicates land on both sides, so the validation score measures memorisation.

Default is to split whole conversations, which also stops the model scoring well
by recognising a voice or a microphone rather than by reading timing.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

SPLIT_KEYS = ("conversation", "boundary", "row")


def group_keys(rows: Sequence[dict], split_by: str = "conversation") -> list[Hashable]:
    """Group key per row, at the requested granularity."""
    if split_by == "conversation":
        return [r.get("conversation_id", "") for r in rows]
    if split_by == "boundary":
        return [(r.get("conversation_id", ""), r.get("speaker", ""), r.get("ipu_end_s")) for r in rows]
    if split_by == "row":
        return list(range(len(rows)))
    raise ValueError(f"split_by must be one of {SPLIT_KEYS}, got {split_by!r}")


def group_split(
    keys: Sequence[Hashable], val_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole groups to train or validation.

    Groups are shuffled and taken until the validation share is met, so the
    split is deterministic given ``seed`` and never straddles a group.
    """
    unique = sorted({str(k): k for k in keys}.keys())
    if len(unique) < 2:
        raise ValueError(
            f"Only {len(unique)} group(s) to split; use a finer split_by or more conversations"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(np.array(unique, dtype=object)))
    target = max(1, round(len(keys) * val_fraction))

    counts: dict[str, int] = {}
    for key in keys:
        counts[str(key)] = counts.get(str(key), 0) + 1

    val_groups: set[str] = set()
    held = 0
    for group in shuffled:
        if held >= target and val_groups:
            break
        val_groups.add(group)
        held += counts[group]

    in_val = np.array([str(k) in val_groups for k in keys])
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)

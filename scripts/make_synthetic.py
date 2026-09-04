"""Synthetic examples in the canonical schema.

This dataset exists to prove the pipeline runs end to end -- ingest shape,
label fields, metrics, baselines, training loop -- not to be a hard problem.
The acoustic cue is deliberately trivial: completed turns go quiet at the end
of the window. Real conclusions come from `scripts/ingest.py` on real audio.

It mirrors the real labeller's structure, though: boundaries are holds or
shifts, and an observation is only emitted while the outcome is still unknown.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from turntiming.data.schema import TimingExample, write_jsonl

OFFSETS = (0.1, 0.3, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--boundaries", type=int, default=800)
    parser.add_argument("--frames", type=int, default=201)
    parser.add_argument("--mels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples: list[TimingExample] = []

    for index in range(args.boundaries):
        turn_end = int(rng.random() > 0.5)
        # Shifts are answered quickly; holds leave a longer, more variable pause.
        gap_s = max(0.05, rng.gauss(0.38, 0.18) if turn_end else rng.gauss(1.10, 0.45))
        ipu_end_s = 5.0 + index * 7.0

        features = [
            [rng.gauss(0, 1) * (0.15 if turn_end and frame > args.frames * 0.8 else 1.0)
             for _ in range(args.mels)]
            for frame in range(args.frames)
        ]

        for offset in OFFSETS:
            if offset > gap_s:  # somebody has already started; nothing left to predict
                break
            examples.append(
                TimingExample(
                    conversation_id=f"syn{index // 50:03d}",
                    speaker="A" if index % 2 else "B",
                    window_end_s=round(ipu_end_s + offset, 4),
                    ipu_end_s=ipu_end_s,
                    observation_offset_s=offset,
                    pause_duration_s=offset,
                    turn_end=turn_end,
                    time_to_next_speaker_s=round(gap_s, 4),
                    next_speaker="B" if index % 2 else "A",
                    features=features,
                )
            )

    n = write_jsonl(Path(args.out), examples)
    positives = sum(e.turn_end for e in examples)
    print(f"wrote {n} examples -> {args.out}  turn_end rate={positives / max(n, 1):.3f}")


if __name__ == "__main__":
    main()

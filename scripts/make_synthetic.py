from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--mels", type=int, default=64)
    args = parser.parse_args()

    rng = random.Random(42)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for _ in range(args.rows):
            turn_end = int(rng.random() > 0.5)
            # Toy signal: completed turns become quieter toward the end of the window.
            features = []
            for frame in range(args.frames):
                scale = 1.0
                if turn_end and frame > int(args.frames * 0.8):
                    scale = 0.15
                features.append([(rng.gauss(0, 1) * scale) for _ in range(args.mels)])
            onset_ms = max(0.0, rng.gauss(380 if turn_end else 1100, 180))
            f.write(
                json.dumps(
                    {
                        "features": features,
                        "turn_end": turn_end,
                        "time_to_next_speaker_ms": onset_ms,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn a real conversational corpus into labelled timing examples.

    python scripts/ingest.py --adapter ami \
        --source data/raw/ami/ami_public_manual_1.6.2 \
        --audio-dir data/raw/ami/amicorpus \
        --out data/ami.jsonl

The output is a manifest: labels plus ``audio_path`` and ``window_end_s``. Mel
features are cut from the audio at load time, so this file stays small enough
to read with your own eyes -- which you should, at least once.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from turntiming.data.adapters import ADAPTERS, ami, load_segments
from turntiming.data.label import LabelConfig, build_examples
from turntiming.data.schema import write_jsonl


def resolve_audio(
    conversation_ids: set[str], audio_dir: Path | None, pattern: str
) -> dict[str, str]:
    """Map conversation ids to audio files, keeping only ones that exist."""
    if audio_dir is None:
        return {}
    found = {}
    for conversation_id in sorted(conversation_ids):
        candidate = audio_dir / pattern.format(conversation_id=conversation_id)
        if candidate.exists():
            found[conversation_id] = str(candidate)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--source", required=True, help="annotation directory or file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conversations", nargs="*", default=None, help="ids to keep")
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument(
        "--audio-pattern",
        default="{conversation_id}/audio/{conversation_id}.Mix-Headset.wav",
        help="path template under --audio-dir",
    )
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="drop conversations with no matching audio file",
    )
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--ipu-gap", type=float, default=0.18)
    parser.add_argument("--min-ipu", type=float, default=0.20)
    parser.add_argument("--max-lookahead", type=float, default=10.0)
    parser.add_argument("--keep-backchannel-shifts", action="store_true",
                        help="treat a backchannel as taking the floor (default: it does not)")
    args = parser.parse_args()

    conversations = args.conversations
    if args.adapter == "ami":
        available = ami.available_meetings(args.source)
        conversations = conversations or available
        unknown = sorted(set(conversations) - set(available))
        if unknown:
            raise SystemExit(f"No AMI annotations for: {', '.join(unknown)}")
    if conversations and args.limit_conversations:
        conversations = sorted(conversations)[: args.limit_conversations]

    kwargs = {"meetings": conversations} if args.adapter == "ami" else {}
    segments = list(load_segments(args.adapter, args.source, **kwargs))
    if conversations and args.adapter != "ami":
        keep = set(conversations)
        segments = [s for s in segments if s.conversation_id in keep]
    if not segments:
        raise SystemExit("Adapter produced no segments")

    ids = {s.conversation_id for s in segments}
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    audio_paths = resolve_audio(ids, audio_dir, args.audio_pattern)
    if args.require_audio:
        segments = [s for s in segments if s.conversation_id in audio_paths]
        if not segments:
            raise SystemExit(f"No audio found under {audio_dir} matching {args.audio_pattern}")

    cfg = LabelConfig(
        ipu_merge_gap_s=args.ipu_gap,
        observation_offsets_s=tuple(args.offsets),
        min_ipu_duration_s=args.min_ipu,
        max_lookahead_s=args.max_lookahead,
        backchannel_max_duration_s=0.0 if args.keep_backchannel_shifts else 1.0,
    )

    stats: Counter = Counter()
    examples = list(build_examples(segments, cfg, audio_paths, stats=stats))
    n = write_jsonl(args.out, examples)

    positives = sum(e.turn_end for e in examples)
    print(f"conversations={len(ids)}  with audio={len(audio_paths)}  segments={len(segments)}")
    print(f"wrote {n} examples -> {args.out}")
    print(f"  turn_end rate     {positives / max(n, 1):.3f}  ({positives} shift / {n - positives} hold)")
    print(f"  backchannels      {sum(e.is_backchannel for e in examples)}")
    print(f"  censored onsets   {sum(e.onset_censored for e in examples)}")
    print("  boundary accounting: " + json.dumps(dict(sorted(stats.items()))))
    for offset in sorted(cfg.observation_offsets_s):
        sub = [e for e in examples if e.observation_offset_s == offset]
        if sub:
            rate = sum(e.turn_end for e in sub) / len(sub)
            print(f"    offset {offset:>4.2f}s  n={len(sub):<6d} turn_end_rate={rate:.3f}")


if __name__ == "__main__":
    main()

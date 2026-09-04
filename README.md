# Conversational Timing Lab

A small research repo for learning **when a speaker has finished a turn** and **when a listener would naturally begin responding**.

The core idea is to model two related but distinct targets:

1. **Turn-end probability** — has the current speaker finished their turn?
2. **Response-onset timing** — how long after the current moment would a human conversational partner naturally begin speaking?

The long-term inference design is latency-aware: the system can begin preparing a response before it is safe to speak, then release it at a predicted natural onset time.

## Why this repo starts simple

This project deliberately begins with compact models trained from scratch rather than a pretrained language/audio foundation model. That makes it easier to understand what the model is actually learning and to run controlled experiments on modest hardware.

The first baseline uses log-mel audio features and a small recurrent encoder. Text, semantic features, speaker context, and multimodal fusion can be added once the timing-only baseline is measurable.

## Data format

Each training example is a short audio window ending at time `t`, with labels derived from conversational annotations.

```json
{
  "audio_path": "clips/dialogue_001.wav",
  "window_end_s": 12.84,
  "speaker_id": "A",
  "turn_end": 1,
  "time_to_next_speaker_ms": 420,
  "next_speaker_id": "B"
}
```

Useful source annotations include speaker turns, timestamps, overlaps, interruptions, and gaps between speakers.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/make_synthetic.py --out data/synthetic.jsonl
python -m turntiming.train --config configs/baseline.yaml --data data/synthetic.jsonl
```

## First experiments

- **E0:** Predict turn end from pause duration only.
- **E1:** Predict turn end from audio features around the current boundary.
- **E2:** Predict response onset delay as a regression target.
- **E3:** Joint model with shared encoder + two heads.
- **E4:** Add transcript/semantic features and test whether they reduce false turn endings on clauses such as “my name is…” followed by a hesitation.
- **E5:** Build a latency-aware simulator that measures *perceived timing error*, not just model accuracy.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the research plan.

## Movie / TV dialogue data

Film and TV dialogue can be useful for prototyping, but subtitles are not reliable enough as ground-truth speech boundaries by themselves. If you use media you legally have access to, align diarized speech/audio boundaries directly and treat subtitle text as an auxiliary signal.

For more publishable experiments, move toward openly licensed conversational speech corpora with speaker-turn timestamps.

## Research question

> Can a small model trained on conversational timing learn both *whether a turn is complete* and *when a response should begin*, well enough to make a voice agent feel less interruptive and less sluggish?

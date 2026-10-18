# conversational-timing-lab

Research repo (`turntiming`) for learning **when a speaker has finished a turn** and
**when a listener would naturally begin responding**. Two related but distinct targets:
turn-end probability, and time-to-response-onset.

## Environment

Python 3.12 in a local venv. **Bare `python3` on this machine is the system 3.9** — always
go through the venv.

```bash
source .venv/bin/activate      # or call .venv/bin/<tool> directly
pytest                         # full suite
ruff check .                   # lint (line-length 100)
```

## Layout

- `src/turntiming/data/` — ingest and labelling (`label.py` is the heart of the repo)
- `src/turntiming/features/`, `models/`, `baselines/`, `eval/`
- `src/turntiming/train.py` — training entry point
- `configs/*.yaml` — experiment configs (`baseline.yaml` synthetic, `ami.yaml` real AMI corpus)
- `scripts/` — `fetch_ami.py`, `ingest.py`, `make_synthetic.py`
- `docs/EXPERIMENTS.md` — **read this before changing labelling or metrics**
- `results/` — committed experiment output

## The three invariants that keep this project honest

`docs/EXPERIMENTS.md` states these; they are the difference between real findings and a
model that has learned to run a voice-activity detector. Do not weaken them for convenience.

1. **Labels come from who speaks next, never from the pause.** A hold and a shift with
   identical 400 ms silences must get opposite labels.
   Pinned by `tests/test_label.py::test_identical_silence_gets_opposite_labels`.
2. **Observations are causal.** Boundaries are observed at 100/300/500/1000 ms after speech
   stops, and only while nobody has spoken yet. `pause_duration_s` is the silence heard *so
   far*, never the silence that eventually elapsed — otherwise the feature encodes the answer.
3. **Contested boundaries are dropped.** If another speaker was already talking when this one
   stopped, there is no pause to interpret (~60% of AMI boundaries).

## Evaluation

The headline safety metric is the **false-interruption rate**, `FIR = FP / (FP + TN)` —
deliberately not precision, because precision moves with the base rate and FIR does not.
Report F1 broken out by observation offset, and the operating point at a fixed interruption
budget (default `max_false_interruption_rate: 0.05`). For onset, report MAE in ms plus the
perceived-timing bands (too early / natural).

## Conventions

- `split_by` matters: split by conversation for real corpora, by boundary only for synthetic
  data where there are too few conversations.
- Keep `results/*.txt` as the record of what was actually run; add to it rather than editing.
- The repo deliberately starts with small models trained from scratch, not a pretrained audio
  foundation model, so that what the model learns stays inspectable. Do not swap in a large
  pretrained backbone without saying why in `docs/EXPERIMENTS.md`.

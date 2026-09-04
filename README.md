# Conversational Timing Lab

A small research repo for learning **when a speaker has finished a turn** and **when a listener would naturally begin responding**.

The core idea is to model two related but distinct targets:

1. **Turn-end probability** — has the current speaker finished their turn?
2. **Response-onset timing** — how long after the current moment would a human conversational partner naturally begin speaking?

The long-term inference design is latency-aware: the system can begin preparing a response before it is safe to speak, then release it at a predicted natural onset time.

## The actual question

Manufacturing a dataset where "quiet = done talking" teaches a model nothing, because that rule is the input. Real conversation contains both of these, with the same silence in the middle:

```
"So what I was thinking was…"     400 ms     "…maybe we should—"      -> still their turn
"Yeah, that sounds good."         400 ms     (other person starts)    -> turn ended
```

Can a tiny model trained from scratch learn the difference? That is the whole project. Everything here exists to make that question measurable on real human conversation.

## Why this repo starts simple

This project deliberately begins with compact models trained from scratch rather than a pretrained language/audio foundation model. That makes it easier to understand what the model is actually learning and to run controlled experiments on modest hardware.

The first baseline uses log-mel audio features and a small recurrent encoder. Text, semantic features, speaker context, and multimodal fusion can be added once the timing-only baseline is measurable.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Check the pipeline on synthetic data — it is a plumbing test, not a result:

```bash
python scripts/make_synthetic.py --out data/synthetic.jsonl
python -m turntiming.train --config configs/baseline.yaml --data data/synthetic.jsonl
```

Then run on real conversation:

```bash
python scripts/fetch_ami.py --out data/raw/ami --meetings ES2002a ES2002b ES2002c ES2002d
python scripts/ingest.py --adapter ami \
    --source data/raw/ami/ami_public_manual_1.6.2 \
    --audio-dir data/raw/ami/amicorpus --require-audio \
    --out data/ami.jsonl
python -m turntiming.baselines.pause --data data/ami.jsonl      # E0, no audio needed
python -m turntiming.train --config configs/ami.yaml --data data/ami.jsonl
```

## Real conversational data

The default corpus is the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) — roughly 100 hours of spontaneous multi-party meetings, released under **CC BY 4.0**, with manually corrected word-level timings per speaker. Word timings matter: they give real speech boundaries rather than the loose ones you get from subtitles or transcriber segments.

`scripts/fetch_ami.py` downloads it. Annotations are a single 23 MB zip covering all 171 meetings; audio is fetched per meeting (~40 MB each), so you only pull what you need. **E0 needs no audio at all** — it runs on timing alone, so you can measure the baseline over the whole corpus in a couple of minutes.

Adding another corpus means writing one function. Adapters turn a corpus's native format into `SpeakerSegment` records, and everything downstream is corpus-agnostic:

| adapter | source |
| --- | --- |
| `ami` | AMI NXT word-level XML annotations |
| `rttm` | any NIST RTTM file or directory — i.e. anything you have already diarised |
| `jsonl` | segments already in the canonical shape |

### On subtitles and TV dialogue

Film and TV dialogue can be useful for prototyping, but subtitles are not reliable enough as ground-truth speech boundaries by themselves. If you use media you legally have access to, align diarized speech/audio boundaries directly, emit RTTM, and treat subtitle text as an auxiliary signal.

## How examples are labelled

`src/turntiming/data/label.py` enforces one rule: **a silence is not a turn end**. Boundaries are labelled by *who speaks next*, never by how quiet it got.

```
segments -> inter-pausal units -> boundaries -> observations
```

- Consecutive speech from one speaker separated by less than `--ipu-gap` (default 180 ms) is one **inter-pausal unit**. Every IPU end is a candidate boundary.
- A boundary is `turn_end=1` if a *different* speaker speaks next, and `turn_end=0` if the same speaker resumes. A speaker who pauses for 800 ms and carries on is the most valuable negative example in the dataset.
- A short backchannel — "mm-hmm", "right" — that the speaker talks straight through does **not** count as taking the floor.
- Boundaries where somebody else was already talking are dropped. There is no pause there to interpret; the floor was contested, not offered.
- Each boundary is observed at several offsets after speech stops (default 100/300/500/1000 ms), and **only while the outcome is still unknown**. Once anyone has started speaking, there is nothing left to predict, so no example is emitted. This is what keeps `pause_duration_s` causal — it is the silence heard *so far*, never the silence that eventually elapsed.

## Data format

One JSON object per decision point. All times are **seconds**; milliseconds appear only in metric reports, where they read better.

```json
{
  "conversation_id": "ES2002a",
  "speaker": "A",
  "window_end_s": 13.24,
  "ipu_end_s": 12.84,
  "observation_offset_s": 0.4,
  "pause_duration_s": 0.4,
  "turn_end": 1,
  "time_to_next_speaker_s": 0.42,
  "next_speaker": "B",
  "is_backchannel": 0,
  "is_overlap": 0,
  "onset_censored": 0,
  "audio_path": "data/raw/ami/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav",
  "text": "so what I was thinking was"
}
```

This is a manifest, not a feature dump: mel windows are decoded from the audio at load time, so the file stays small enough to read with your own eyes — which you should, at least once. Synthetic examples carry inline `features` instead, and `build_dataset` picks the right loader either way.

`onset_censored` marks boundaries where nobody spoke again within the lookahead horizon. The onset target there means "further away than 10 s", so it is excluded from the regression loss and from MAE.

## Evaluation

Accuracy is close to useless here, because the two errors are not symmetric. Speaking when the user had not finished is an interruption, and interruptions are what make an agent unbearable to talk to.

So `turntiming.eval.metrics` reports precision, recall and F1, plus:

- **False-interruption rate**, `FIR = FP / (FP + TN)` — of all the moments the user was *still going*, how often the agent would have barged in. Precision answers "when it spoke, was it right?"; FIR answers "how often did it talk over me?", and only the second is what users complain about.
- **Metrics at a fixed interruption budget** — the operating point a real agent ships at. Pick an FIR you can live with, then take all the recall it buys you.
- **A per-offset breakdown**, since the problem changes shape with how long you have waited.
- **Perceived timing bands** for response onset — interruptive / natural / slightly late / sluggish — alongside MAE in milliseconds.

## Experiments

- **E0:** Pause-duration threshold and majority class. The baseline everything else must beat. `python -m turntiming.baselines.pause`
- **E1:** Predict turn end from audio features around the current boundary.
- **E2:** Predict response onset delay as a regression target.
- **E3:** Joint model with shared encoder + two heads.
- **E4:** Add transcript/semantic features and test whether they reduce false turn endings on clauses such as "my name is…" followed by a hesitation.
- **E5:** Build a latency-aware simulator that measures *perceived timing error*, not just model accuracy.

Every training run prints E0 on the same validation split, because a timing model that cannot beat "wait for silence" has not earned its parameters.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the research plan, and [`results/`](results/) for verbatim run output.

### What E0 found

On all 171 AMI meetings (100,004 boundaries, conversation-level split), the pause-duration baseline learns this:

```
P(turn ended | silence so far):  100ms -> 0.633   300ms -> 0.589   500ms -> 0.543   1000ms -> 0.435
```

It runs **backwards**. The longer the silence has lasted, the *less* likely the speaker was finished — a speaker thinking mid-turn out-waits a partner simply taking their next turn. The best pause-only model beats the majority class by 3.5 accuracy points, and none of them can hit a 5% interruption budget at any threshold. That is the bar E1 has to clear. Full numbers in [`results/`](results/README.md).

## Research question

> Can a small model trained on conversational timing learn both *whether a turn is complete* and *when a response should begin*, well enough to make a voice agent feel less interruptive and less sluggish?

## Citation

If you publish work using the AMI data, cite the corpus:

> Carletta, J. et al. (2006). *The AMI Meeting Corpus: A Pre-Announcement.* Machine Learning for Multimodal Interaction. AMI is distributed under CC BY 4.0.

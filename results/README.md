# Initial results

Verbatim run output, one file per run. Each file records the command and the
commit that produced it. Regenerate any of them by re-running the command in
the header — nothing here is hand-edited.

Getting the data first:

```bash
python scripts/fetch_ami.py --out data/raw/ami --annotations-only   # 23 MB, all 171 meetings
```

| file | what it is |
| --- | --- |
| [`ingest_ami_full.txt`](ingest_ami_full.txt) | Dataset statistics for all 171 AMI meetings |
| [`e0_ami_full.txt`](e0_ami_full.txt) | **E0 baselines on 100,004 real boundaries** |
| [`e1_ami_3meetings.txt`](e1_ami_3meetings.txt) | E1 pipeline check on real audio — *not a result* |

## What E0 says

The pause→rate lookup learns this conditional, on a conversation-level split:

```
P(turn ended | silence so far):  100ms -> 0.633   300ms -> 0.589   500ms -> 0.543   1000ms -> 0.435
```

It runs **backwards**. The longer the silence has lasted, the *less* likely the
speaker was finished — because a speaker who is merely thinking mid-turn
out-waits a partner who is simply taking their next turn. Every silence-threshold
endpointer assumes the opposite sign.

| model | accuracy | P | R | F1 | FIR |
| --- | --- | --- | --- | --- | --- |
| E0a majority class | 0.538 | 0.538 | 1.000 | 0.700 | 1.000 |
| E0b pause threshold (shorter than 300 ms) | 0.567 | 0.588 | 0.651 | 0.618 | 0.531 |
| E0c pause → rate lookup | 0.573 | 0.567 | 0.877 | 0.688 | 0.781 |

Two things to carry into E1:

- The best pause-only model beats the majority class by **3.5 accuracy points**.
  Pause duration is close to worthless for this decision.
- **No pause-only model can hit a 5% interruption budget at all.** Its
  probabilities never leave the 0.43–0.63 band, so no threshold buys both
  non-zero recall and a low false-interruption rate. Producing a score that
  separates far enough to be usable at a real operating point is E1's job.

The threshold baseline is fitted on Youden's J rather than F1. Fitted on F1 it
degenerates into "always predict turn ended" — with a 0.561 base rate that is
the F1 optimum — and reports 0.700 F1 while having learned nothing. Worth
knowing before trusting any F1 number on this dataset.

## What E1 does *not* say yet

`e1_ami_3meetings.txt` trains on 643 examples from two meetings and validates on
one, which is not enough data to conclude anything. It lands at 0.552 accuracy —
*below* the 0.571 the majority class gets on that split, and below E0's 0.569 —
with validation F1 swinging between 0.24 and 0.59 across epochs. That is a
statement about data volume, not about the model.

The file exists to show the audio path runs end to end: ingest → frame-offset
window decoding → training → evaluation against E0 on the same conversation-level
split. Every number in it should be ignored.

A real E1 needs audio for many more meetings (~40 MB each):

```bash
python scripts/fetch_ami.py --out data/raw/ami --meetings ES2002a ES2002b ...
python scripts/ingest.py --adapter ami --source data/raw/ami/ami_public_manual_1.6.2 \
    --audio-dir data/raw/ami/amicorpus --require-audio --out data/ami.jsonl
python -m turntiming.train --config configs/ami.yaml --data data/ami.jsonl
```

## Caveats

- AMI is four-party meeting speech. 60% of boundaries are dropped as contested
  (someone else was already talking), and meeting turn-taking is not dyadic
  turn-taking. Add a two-party corpus before concluding anything about voice
  agents — the RTTM adapter takes anything you have diarised.
- Onset targets are censored at 10 s; 17,832 of 100,004 examples hit that
  horizon and are excluded from onset MAE.

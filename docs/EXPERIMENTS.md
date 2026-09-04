# Experiment plan

## Objective

Separate conversational timing into two prediction problems:

- **Turn completion:** probability that the active speaker has yielded the floor.
- **Response onset:** predicted time until the other speaker should begin speaking.

The second target is not merely "silence duration". It captures conversational rhythm: immediate acknowledgements, ordinary gaps, deliberate pauses, hesitation, overlap, and interruption.

## Labels

For each sample time `t` during speaker A's turn:

- `turn_end = 1` if A's current conversational turn has ended by `t`.
- `time_to_next_speaker_s = max(0, onset(B) - t)` for the next non-A speech onset.
- also emitted: `is_overlap`, `is_backchannel`, `onset_censored`, `pause_duration_s`, `observation_offset_s`.

Do not force every silence to mean "turn ended". A pause followed by the same speaker is a particularly valuable negative example.

Implemented in `src/turntiming/data/label.py`. Three decisions there are worth stating explicitly, because each one is a way the dataset could quietly become trivial:

1. **Labels come from who speaks next, never from the pause.** A hold and a shift with identical 400 ms silences must get opposite labels, or the model is only being taught to run a VAD. `tests/test_label.py::test_identical_silence_gets_opposite_labels` pins this.
2. **Observations are causal.** A boundary is observed at 100/300/500/1000 ms after speech stops, and only while nobody has spoken yet. `pause_duration_s` is therefore the silence heard *so far*, never the silence that eventually elapsed — otherwise the pause feature encodes the answer.
3. **Contested boundaries are dropped.** If another speaker was already talking when this one stopped, there is no pause to interpret. In AMI these are 60% of boundaries; in two-party audio they will be far fewer.

## Evaluation

Implemented in `src/turntiming/eval/metrics.py`.

### Turn end

Use precision/recall/F1 in addition to accuracy. False positives matter disproportionately because they correspond to interrupting the user.

The headline safety number is the **false-interruption rate**, `FIR = FP / (FP + TN)`: of all the moments the user was still going, how often the agent would have barged in. It is deliberately not precision — precision moves with the base rate, and FIR does not.

Because the base rate shifts with how long you have waited, every report also breaks out F1 by observation offset (100, 300, 500, 1000 ms), and reports the operating point at a fixed interruption budget (default FIR ≤ 5%), which is how an agent actually ships.

### Response onset

Report MAE in milliseconds, but also evaluate perceived timing bands:

- too early / interruptive (predicted more than 200 ms early)
- natural (within 200 ms)
- slightly late (200–600 ms)
- noticeably sluggish (more than 600 ms late)

Censored targets — nobody spoke again within the lookahead horizon — are excluded from the loss and from MAE, since they only mean "further away than 10 s".

A useful application metric is **release timing error** after subtracting the measured generation/TTS latency of the agent stack.

## Suggested progression

### E0 — heuristics *(done)*

`python -m turntiming.baselines.pause --data data/ami_all.jsonl`

Three baselines: majority class, a fitted pause threshold, and a pause→turn-end-rate lookup table. The threshold fits **both directions** and is fitted on Youden's J rather than F1, because with an unbalanced base rate F1 is maximised by predicting the majority class everywhere, and a baseline that silently degenerates proves nothing.

**Result on all 171 AMI meetings (100,004 boundaries, conversation-level split):**

| model | accuracy | P | R | F1 | FIR |
| --- | --- | --- | --- | --- | --- |
| majority class | 0.538 | 0.538 | 1.000 | 0.700 | 1.000 |
| pause threshold (shorter than 300 ms) | 0.567 | 0.588 | 0.651 | 0.618 | 0.531 |
| pause → rate lookup | 0.573 | 0.567 | 0.877 | 0.688 | 0.781 |

The learned conditional is the point of the whole exercise:

```
P(turn ended | silence so far):  100ms -> 0.633   300ms -> 0.589   500ms -> 0.543   1000ms -> 0.435
```

It runs **backwards**. The longer the silence has lasted, the *less* likely the speaker was finished — because a speaker who is merely thinking mid-turn out-waits a partner who is simply taking their next turn. Every silence-threshold endpointer assumes the opposite.

Two consequences worth carrying into E1:

- The best pause-only model beats the majority class by 3.5 accuracy points. Pause duration is close to worthless here, and the synthetic dataset's premise — quiet means done — is not merely a simplification but the wrong sign.
- **No pause-only model can hit a 5% interruption budget at all.** Its probabilities never leave the 0.43–0.63 band, so no threshold buys both non-zero recall and a low FIR. That is the bar: E1's job is to produce a score that separates far enough to be usable at a real operating point.

### E1 — small audio model

Train the GRU baseline from scratch on 1–3 seconds of log-mel context.

Splits are by conversation (`split_by: conversation`), so the model cannot score well by recognising a voice or a room. The synthetic config splits by boundary instead, since it has few conversations.

### E2 — joint timing model

Train turn-end classification and onset-delay regression jointly. Compare against two independently trained models.

### E3 — semantic branch

Add a tiny text encoder trained from scratch over characters/subwords from transcripts. Test examples such as incomplete clauses, hesitation, lists, and self-correction. `text` is already carried on every example for this.

### E4 — conversational history

Condition on the previous few turns, speaker identity, previous gap lengths, and whether the current utterance looks like a question, acknowledgement, or continuation.

### E5 — live simulator

Measure actual ASR + model + LLM + TTS latency. Begin speculative response generation before the predicted onset, but gate audible speech until a release policy says it is safe.

A basic release rule could be:

`release_at = predicted_response_onset - downstream_audio_start_latency`

Cancel or revise speculative generation if the speaker resumes.

## Dataset strategy

Start with a small, manually inspectable dataset. A few thousand high-quality boundaries can teach you more than millions of noisy subtitle timestamps.

The default corpus is AMI (CC BY 4.0), with word-level manual timings. `scripts/ingest.py` turns any adapter's segments into examples; adding a corpus means writing one function that yields `SpeakerSegment` records.

Create hard negatives intentionally:

- speaker pauses mid-sentence
- "uh", "um", and repair sequences
- sentence completion followed by immediate continuation
- rhetorical questions
- backchannels
- overlap and interruption
- slow speakers and fast speakers

The labeller already surfaces several of these directly: `is_backchannel` marks the "mm-hmm" case, `is_overlap` marks talked-over turns, and every hold is by construction a pause that did not end a turn.

Synthetic examples are useful for testing the pipeline and balancing edge cases, but final conclusions should be based on real conversational audio.

### Known limitations

- AMI is four-party meeting speech. It is real and spontaneous, but the turn-taking dynamics of a meeting are not those of a two-party conversation, and 60% of boundaries are dropped as contested. A two-party corpus should be added before drawing conclusions about dyadic voice agents.
- `Mix-Headset` is a single mixed channel. Per-speaker headset tracks are available and would let the model hear one speaker cleanly.
- Backchannel detection is a lexicon plus a duration rule, not a classifier.

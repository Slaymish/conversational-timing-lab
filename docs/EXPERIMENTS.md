# Experiment plan

## Objective

Separate conversational timing into two prediction problems:

- **Turn completion:** probability that the active speaker has yielded the floor.
- **Response onset:** predicted time until the other speaker should begin speaking.

The second target is not merely "silence duration". It captures conversational rhythm: immediate acknowledgements, ordinary gaps, deliberate pauses, hesitation, overlap, and interruption.

## Labels

For each sample time `t` during speaker A's turn:

- `turn_end = 1` if A's current conversational turn has ended by `t`.
- `time_to_next_speaker_ms = max(0, onset(B) - t)` for the next non-A speech onset.
- optionally: `is_overlap`, `is_interruption`, `same_speaker_resumes`, `backchannel`.

Do not force every silence to mean "turn ended". A pause followed by the same speaker is a particularly valuable negative example.

## Evaluation

### Turn end

Use precision/recall/F1 in addition to accuracy. False positives matter disproportionately because they correspond to interrupting the user.

Also report F1 at several observation offsets, such as 100, 300, 500, and 1000 ms after speech cessation.

### Response onset

Report MAE in milliseconds, but also evaluate perceived timing bands:

- too early / interruptive
- natural
- slightly late
- noticeably sluggish

A useful application metric is **release timing error** after subtracting the measured generation/TTS latency of the agent stack.

## Suggested progression

### E0 — heuristics

Pause-duration threshold and logistic regression on simple acoustic statistics. This is the baseline everything else must beat.

### E1 — small audio model

Train the GRU baseline from scratch on 1–3 seconds of log-mel context.

### E2 — joint timing model

Train turn-end classification and onset-delay regression jointly. Compare against two independently trained models.

### E3 — semantic branch

Add a tiny text encoder trained from scratch over characters/subwords from transcripts. Test examples such as incomplete clauses, hesitation, lists, and self-correction.

### E4 — conversational history

Condition on the previous few turns, speaker identity, previous gap lengths, and whether the current utterance looks like a question, acknowledgement, or continuation.

### E5 — live simulator

Measure actual ASR + model + LLM + TTS latency. Begin speculative response generation before the predicted onset, but gate audible speech until a release policy says it is safe.

A basic release rule could be:

`release_at = predicted_response_onset - downstream_audio_start_latency`

Cancel or revise speculative generation if the speaker resumes.

## Dataset strategy

Start with a small, manually inspectable dataset. A few thousand high-quality boundaries can teach you more than millions of noisy subtitle timestamps.

Create hard negatives intentionally:

- speaker pauses mid-sentence
- "uh", "um", and repair sequences
- sentence completion followed by immediate continuation
- rhetorical questions
- backchannels
- overlap and interruption
- slow speakers and fast speakers

Synthetic examples are useful for testing the pipeline and balancing edge cases, but final conclusions should be based on real conversational audio.

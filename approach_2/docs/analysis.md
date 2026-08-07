# Analysis: Consensus-Based STT Quality Assessment (Approach 2)

This document analyzes the proposed **two-engine consensus** pipeline. It separates
what is sound from what needs precision, so the plan in `plan.md` is unambiguous
enough for a mid-level engineer to implement without guessing.

---

## 1. What the approach gets right

1. **No human ground truth required.** Agreement between two independent STT
   engines is a production-viable proxy for correctness. This is the whole point
   of "Approach 2" in `../approaches.md`.
2. **Agreement is a strong, deterministic error signal.** When two engines
   disagree word-for-word, the segment is where the audio is hard to hear — the
   exact places a human should verify.
3. **Risk-tiered review is efficient and statistically defensible.** Reviewing
   *all* low-confidence + *all* disagreement segments, then a seeded random 10% of
   the rest, concentrates human effort where errors are most likely while still
   validating the parts both engines agree on.
4. **Thresholds with explicit actions** (98–100 auto-accept, 90–97 conditional,
   <90 mandatory) are clear, configurable, and auditable.
5. **Timestamp-anchored spot checks** keep review proportional to duration and
   are measurable: an acceptance criterion (≥99% on the sample) turns QA into a
   pass/fail gate instead of a judgment call.

---

## 2. Where the approach is underspecified (must be tightened)

### 2.1 Alignment is the crux, and naive approaches fail

The stated "align segments using timestamps, else text alignment" is the right
instinct but hides real complexity. In practice:

- Engines **split sentences differently** (one segment vs three).
- Engines **merge** multiple sentences into one segment.
- Timestamps **drift**: boundaries are never equal; a constant offset can exist
  across the whole file.
- Engines **omit filler words** and punctuation, shifting every word position.

Consequences that make naive line-by-line comparison wrong:

- Timestamps alone **cannot** decide matching — overlap is a *heuristic*, not a
  truth. Two segments that overlap in time may still be different text.
- Text alone **cannot** decide matching either — without timestamps you cannot
  disambiguate a reordering from a deletion.
- 1:1 pairing is insufficient: an engine that merges three sentences must be
  aligned to all three of the other engine's segments (many-to-one).

**Decision for the plan:** flatten both engines to normalized word streams and
run one global Needleman–Wunsch over word tokens (equal words +1, gaps
substitutions −1), penalizing a diagonal match when its interpolated timestamps
drift beyond tolerance so a repeated phrase cannot be smeared across a dropped
span. Regroup the aligned words into per-segment windows (merging adjacent
windows that a single other-engine segment spans), so each comparison covers the
same spoken content regardless of how each engine cut it. This avoids
segment-boundary matching entirely — the source of the 1:N failure — instead of
patching it with a post-pass.

### 2.2 "Compare aligned segments" needs a defined diff, not eyeballing

Highlighting insertions/deletions/substitutions and computing *agreement* both
require a **word-level edit alignment** (Levenshtein over word tokens, per
aligned segment pair). From that diff:

- `agreement = 1 - WER(norm(A), norm(B))` becomes a deterministic, per-segment
  number.
- Each word gets an edit operation (`match | substitute | insert | delete`),
  which is exactly what the review UI needs to highlight.

### 2.3 The confidence score must be a formula, not a vibe

"Combine engine confidence + agreement + low-confidence words + domain terms" is
not implementable as written. The plan locks a deterministic formula with
explicit inputs and fixed weights (see `plan.md §5.5`).

### 2.4 Engine confidence signals are not directly comparable

- Faster-Whisper returns per-segment `avg_logprob` and per-word token
  probabilities; segment confidence is derived as `exp(avg_logprob)`.
- Deepgram Nova returns per-word and per-utterance confidence in `[0, 1]`.

The plan must map whisper scores into `[0, 1]` before combining. This is flagged
as a calibration risk (see `§4`), not something to hand-wave.

### 2.5 The spot check needs a sampling rule and an acceptance rule

"Randomly sample ~10%" is underspecified. The plan defines:

- **Sampling:** tier 0 = all segments with confidence < 90; tier 1 = all segments
  where engines disagree; tier 2 = seeded random 10% of the remainder. A fixed
  seed makes the sample reproducible.
- **Acceptance:** the reviewer marks each sampled word correct/incorrect against
  the audio; if sample accuracy ≥ 99% accept; else double the sample and re-check;
  if still below, flag the transcript for full manual review.

### 2.6 Audio preprocessing should be conservative

Volume normalization + 16 kHz mono WAV conversion are mandatory (both engines need
them). Noise reduction is **optional and off by default** — it can degrade clean
audio and is hard to tune.

---

## 3. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Primary engine | Faster-Whisper (local) | Already a dependency; configurable model size |
| Secondary engine | Deepgram Nova (cloud REST) | Needs `DEEPGRAM_API_KEY`; tests use synthetic fixtures, no mock engine |
| Alignment | Global word-level Needleman–Wunsch over normalized word streams (time-penalized diagonals) → window reassembly | Handles split/merge/timestamp drift; no segment-boundary matching |
| Agreement metric | `1 - WER` over normalized words | Deterministic, matches word diff output |
| Confidence formula | `0.40·engine_conf + 0.45·agreement + 0.15·(1 − low_conf_ratio)`; glossary escalation handled by the tier rule | Explicit, weighted, fixed weights |
| Review tiers | `≥98` auto-accept · `90–97` review-if-technical · `<90` mandatory | Matches the user's thresholds |
| Spot check | tier 0 + tier 1 mandatory; seeded 10% of remainder; acceptance ≥99% | Statistically grounded, reproducible |
| LLM judge | Disagreement-only, audio-grounded arbitration (Gemini 3.5 Flash): `classification` + `severity` + per-engine error flags; `LLMJudge` protocol, swappable | Cheap two-engine filter; LLM only where engines disagree |
| Exports | JSON, TXT, MD, SRT, VTT (v1); DOCX, PDF (v2) | Text formats need no extra deps |

---

## 4. Risks and open questions

1. **Deepgram is a cloud API with a key.** `DEEPGRAM_API_KEY` is required to
   transcribe; it is read from the repo root `.env`. Tests never call Deepgram
   (they use synthetic `EngineSegment` fixtures).
2. **Whisper confidence calibration.** Mapping `avg_logprob` to `[0,1]` is a
   heuristic. Plan: logistic mapping as the default, documented as needing
   calibration on a small labeled set. If uncalibrated, the confidence *formula*
   is still meaningful because agreement dominates (0.45 weight).
3. **Filler-word list is domain/language dependent.** Start with a common English
   list as a module constant. Filler words are stripped only for alignment and
   agreement, never removed from the stored transcript.
4. **Domain glossary source is undefined.** v1 accepts a plain wordlist file
   (optional, empty by default). If absent, no terminology escalation happens.
5. **UI scope.** A build-tooled SPA is over-engineering. v1 is one static HTML
   page served by FastAPI; audio playback seeks the `<audio>` element to the
   segment's time span (browser-native, no server-side cutting).
6. **Human corrections persistence.** v1 writes corrections to a JSON sidecar;
   no database.

---

## 5. Summary

The approach is fundamentally sound and production-appropriate. The two things
that would sink it if implemented naively are (a) segment alignment and (b) a
confidence score with no defined formula. The plan resolves both deterministically
and keeps the human in the loop only where risk justifies it.

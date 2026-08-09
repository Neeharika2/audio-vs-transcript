# Evaluation Approaches

The repo contains two working approaches for checking that a transcript matches
its audio. They differ in what they treat as the "correct answer".

## Approach 1 — Compare against a gold transcript (`approach_1/`)

A human reference transcript is the ground truth. Whisper transcribes the audio,
then the two texts are compared and classified into **missing**, **incorrect**,
**conflicting**, or **hallucinated** information. Quick checks handle the easy
cases; a Gemini LLM judge arbitrates the rest.

Needs: a manual reference transcript. Best for benchmarking, where the cost of a
human reference is acceptable.

## Approach 2 — Two-engine consensus (`approach_2/`)

No reference transcript. Whisper and Deepgram transcribe the same audio, their
word streams are aligned, and each segment gets an agreement score. When the
engines agree, the segment is accepted; when they disagree (or a critical signal
fires — a changed number, a flipped negation), a Gemini judge **listens to the
audio clip** and decides what was actually said.

Best for: production monitoring, where manual verification of every file is too
expensive.
# Approach 2 — two-engine consensus + audio-grounded LLM verification

No gold transcript needed: two independent STT engines (Whisper + Deepgram)
transcribe the same audio, their segments are aligned at word level, and each
span is scored by how strongly the engines agree. Tiers decide what a human
must review, and an interactive UI plays the exact audio span for every flagged
segment.

Segments where the two engines **disagree** (or one side is missing) are sent
to an **audio-capable LLM** (Gemini 3.5 Flash) that listens to the actual audio
clip and arbitrates: what was really spoken, which engine erred, and how severe
the error is. The STT engines are the cheap first-pass filter — high-agreement
segments bypass the LLM entirely.

## Setup

```bash
source .venv/bin/activate
cd approach_2
pip install -r requirements.txt
```

`DEEPGRAM_API_KEY` is required (repo-root `.env`) for the secondary engine.
`GEMINI_API_KEY` is required for the LLM judge stage. Whisper runs locally via
`faster-whisper`.

## Config (via `.env`, repo root)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | faster-whisper size (engine A) |
| `DEEPGRAM_API_KEY` | — | required for Deepgram Nova (engine B) |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model |
| `GLOSSARY_PATH` | (empty) | optional domain-term wordlist (one per line) |
| `GEMINI_API_KEY` | — | required for the LLM judge stage |
| `GEMINI_MODEL` | `gemini-3.5-flash` | audio-capable judge model |

## Run

```bash
python -m approach_2.main transcribe            # transcribe all files in dataset/audio
python -m approach_2.main transcribe audio-3.ogg  # one file
python -m approach_2.main review                # evaluate stored transcripts -> reports
python -m approach_2.main review audio-1        # one file
python -m approach_2.main evaluate audio-1      # transcribe (if missing) + review, one step
python -m approach_2.main judge audio-1         # + LLM-judge disagreement segments
```

`transcribe` writes each engine's output to `dataset/<engine>/`:

```
dataset/whisper/<name>.txt            full transcript
dataset/whisper/<name>.segments.json  timestamped segments
dataset/deepgram/<name>.txt
dataset/deepgram/<name>.segments.json
```

`review` (and `evaluate`) run align → compare → score → sample and write a
report to `dataset/review/<name>/`:

```
report.json  report.txt  report.md  report.srt  report.vtt
```

`judge` additionally runs the LLM over disagreement segments and attaches a
structured verdict to each: `classification` (missing/extra/hallucinated/
incorrect/conflicting/accurate), `correct_content`, `whisper_error`,
`deepgram_error`, `severity` (low/medium/high/critical), `explanation`,
`evidence`. Verdicts persist to `dataset/review/<name>/judgments.json`, and
exports use the corrected content when the judge provides it.

## Interactive review

```bash
uvicorn approach_2.api:app --port 8000
```

The single static page (`static/review.html`) lists every segment with its
confidence, tier, and word diff, plays the exact span in the browser, and posts
verdicts/corrections back to `POST /review/<name>`. Verdicts persist to
`dataset/review/<name>/verdicts.json`; acceptance (≥99% on the reviewed sample)
is recomputed on every load.

## How alignment works

Raw segment boundaries are unreliable — one engine splits a sentence into
several segments while the other merges several sentences into one. Instead of
comparing raw segments, both engines are flattened to normalized word streams
and aligned globally with Needleman-Wunsch (order is identical even when
boundaries are not; matches far apart in time are penalized). The aligned words
are grouped back into segments, so every comparison covers the same spoken
content regardless of how each engine happened to cut it. Word-level agreement
(`1 − WER`) is computed only after these spans are aligned.

## LLM judge (disagreement arbitration)

The deterministic agreement score is the first-pass filter. `src/judge.py`
flags a segment for LLM verification only when one engine missed it entirely or
the engines agree below `DISAGREE_THRESHOLD` (0.9). For each flagged segment it
cuts the exact audio span, sends it to Gemini 3.5 Flash together with both
transcripts + timestamps + confidence, and stores the structured verdict. The
`LLMJudge` protocol keeps the model swappable; `GeminiJudge` is the current
implementation.

## Tiers

| Confidence | Tier | Action |
|---|---|---|
| ≥ 98 | auto_accept | accept automatically |
| 90–97 | review_technical | review only if a glossary term is present |
| < 90 | mandatory | must be human-verified |

## Tests

```bash
source .venv/bin/activate
python -m pytest approach_2/tests -q
```

`tests/test_e2e.py` runs the whole path against the committed dataset
transcripts (skipped automatically if they are not present).

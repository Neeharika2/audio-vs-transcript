# Approach 2 — two-engine consensus quality assessment

No gold transcript needed: two independent STT engines (Whisper + Deepgram)
transcribe the same audio, their segments are aligned at word level, and each
span is scored by how strongly the engines agree. Tiers decide what a human
must review, and an interactive UI plays the exact audio span for every flagged
segment.

## Setup

```bash
source .venv/bin/activate
cd approach_2
pip install -r requirements.txt
```

`DEEPGRAM_API_KEY` is required (repo-root `.env`). Whisper runs locally via
`faster-whisper`.

## Config (via `.env`, repo root)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | faster-whisper size (engine A) |
| `DEEPGRAM_API_KEY` | — | required for Deepgram Nova (engine B) |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model |
| `GLOSSARY_PATH` | (empty) | optional domain-term wordlist (one per line) |

## Run

```bash
python -m approach_2.main transcribe            # transcribe all files in dataset/audio
python -m approach_2.main transcribe audio-3.ogg  # one file
python -m approach_2.main review                # evaluate stored transcripts -> reports
python -m approach_2.main review audio-1        # one file
python -m approach_2.main evaluate audio-1      # transcribe (if missing) + review, one step
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

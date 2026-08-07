# Audio vs Transcript — Quality Assessment

Two independent approaches to assessing speech-to-text quality, sharing a common
repo. `approaches.md` explains why both exist.

| | Approach 1 | Approach 2 |
|---|---|---|
| Needs a gold transcript? | Yes (audio + human transcript) | No (audio only) |
| How accuracy is judged | Whisper vs gold, LLM-judged | Whisper vs Deepgram consensus |
| Deterministic core | Signals + scoring | Alignment + agreement + tiers |
| Human review | Error report | Interactive spot-check UI |

**Which one should I use?**
- **Approach 2** (two-engine consensus) when you *only* have audio and want to
  know, per segment, how confident both engines are — no gold transcript needed.
- **Approach 1** when you already have a trusted reference transcript and need
  to audit the STT output against it (e.g. verifying a dictation against source).

If you can get a gold transcript, Approach 1 gives a stronger verdict; Approach 2
is the choice for fully automated, reference-free QA. The two can be combined:
run Approach 2 to flag risky spans, then verify them with Approach 1 if you also
have the transcript.

---

## Approach 1 — gold-reference comparison

Whisper transcribes the audio; the transcript is compared against a gold
reference (`.pdf`/`.txt`) and graded for missing / incorrect / conflicting /
hallucinated information.

```bash
cd approach_1
python -m venv .venv && source .venv/bin/activate   # or reuse the repo venv
pip install -r requirements.txt
cp ../.env .env        # provides DEEPSEEK_API_KEY
```

### Config (via `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `STT_MODEL_NAME` | `base` | faster-whisper size |
| `EVAL_MODEL_NAME` | `deepseek-v4-flash` | LLM judge model |
| `DEEPSEEK_API_KEY` | — | required for the judge |
| `EMBEDDINGS_ENABLED` | `true` | use sentence-transformers signal |
| `SCORE_THRESHOLD` | `90` | below this -> status `Mismatch` |

### Run

```bash
python -m approach_1.main evaluate audio.wav --gold gold.txt
python -m approach_1.main evaluate-text --gold gold.txt --candidate candidate.txt
uvicorn approach_1.api:app --reload    # POST /evaluate (audio+transcript), POST /evaluate-text
```

---

## Approach 2 — two-engine consensus

Two independent STT engines transcribe the same audio; their segments are
aligned (time + text), word-diffed, and scored. Tiers decide what a human must
review, and an interactive UI plays the exact span for each flagged segment.

```bash
source .venv/bin/activate
cd approach_2
pip install -r requirements.txt
```

### Config (via `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | faster-whisper size (engine A) |
| `DEEPGRAM_API_KEY` | — | required for Deepgram Nova (engine B) |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model |
| `GLOSSARY_PATH` | (empty) | optional domain-term wordlist (one per line) |

### Run

```bash
python -m approach_2.main transcribe            # transcribe all files in dataset/audio
python -m approach_2.main transcribe audio-3.ogg  # one file
python -m approach_2.main review                # evaluate stored transcripts -> reports
python -m approach_2.main review audio-1        # one file
python -m approach_2.main evaluate audio-1      # transcribe (if missing) + review in one step
```

Interactive review (play segments, mark correct/incorrect, correct text):

```bash
uvicorn approach_2.api:app --port 8000
```

Verdicts persist to `dataset/review/<name>/verdicts.json`; acceptance (≥99% on
the reviewed sample) is recomputed live. CLI equivalents print the same status.
See `approach_2/README.md` for the data layout, alignment notes, and tests.

### Tiers

| Confidence | Tier | Action |
|---|---|---|
| ≥ 98 | auto_accept | accept automatically |
| 90–97 | review_technical | review only if a glossary term is present |
| < 90 | mandatory | must be human-verified |

---

## Tests

```bash
source .venv/bin/activate
python -m pytest approach_1/tests approach_2/tests -q
```

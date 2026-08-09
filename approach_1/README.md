# Approach 1 — Gold-Reference Transcript Audit

Whisper transcribes the audio, then the transcript is audited against a trusted
**gold reference** (`.pdf`/`.txt`) for **missing / incorrect / conflicting /
hallucinated** information. An LLM judge arbitrates the discrepancies; a
deterministic signal layer keeps the scoring cheap, reproducible, and explainable.

Use this approach when you already have a reference transcript and need a verdict
on how faithfully the STT captured it. For a reference-free (audio-only) audit,
see Approach 2 — the consensus of two engines.

---

## How it works (pipeline)

```
 audio ──► Whisper STT ─────► candidate transcript
                                    │
 gold (.pdf/.txt) ─────────►       ▼
                           segment + align  (sentence / window chunks)
                                    │
                          ┌─────────┴──────────┐
                          │  deterministic     │  signals: WER, CER, coverage,
                          │  signals           │  hallucination ratio, entity
                          │                    │  rec/prec, semantic similarity
                          └─────────┬──────────┘
                                    ▼
                          LLM classify: per-pair relationship
                          (match / incorrect / conflict / missing / hallucination)
                                    │
                          global review pass (prune false positives,
                          catch cross-segment contradictions)
                                    ▼
                          score + status (Match / Mismatch)
```

- **Align** (`src/align.py`): splits gold & candidate into sentences (or
  fixed-width windows if the STT has no punctuation) and matches them with
  fuzzy char + optional embedding similarity.
- **Classify** (`src/classify.py`): each aligned pair is judged by a small LLM
  prompt; unmatched segments map deterministically to `missing` / `hallucinated`.
  A final global-review pass prunes false positives. Without an API key the
  pipeline falls back to deterministic heuristics and still runs offline.
- **Signals & score** (`src/signals.py`, `src/score.py`): WER, CER, coverage,
  hallucination ratio, entity recall/precision and (optional) semantic
  similarity are combined into a 0–100 score; severity-weighted findings are
  subtracted. Below `SCORE_THRESHOLD`, or any high-severity finding, => `Mismatch`.

## Directory layout

```
approach_1/
├── main.py            CLI entry points
├── api.py             FastAPI server (upload UI + POST endpoints)
├── compare.py         interactive runner (pick audio + transcript, get a report)
├── config.py          .env-driven settings + lazy factories
├── static/index.html  single-page review UI (served by api.py at /)
├── datasets/
│   ├── recordings/            input audio (mp4, ogg, …)
│   ├── manual_transcripts/    gold references (pdf)
│   └── stt_generated_transcripts/   cached Whisper output (<stem>_stt.txt)
├── docs/               architecture / evaluation plan
├── src/
│   ├── evaluate.py     pipeline orchestrator
│   ├── align.py        segmentation + alignment
│   ├── classify.py     LLM (or heuristic) classification
│   ├── signals.py      deterministic signals + sentence embedder
│   ├── score.py        final score + status
│   ├── normalize.py    text normalization for lexical signals
│   ├── evaluator.py   Whisper STT + DeepSeek LLM judge
│   └── models.py       Pydantic response models
└── tests/              pytest suite
```

---

## Setup

Python 3.11+ is assumed. You can reuse the repo-root virtual env.

```bash
cd approach_1

# option A: reuse the repo's .venv
source ../.venv/bin/activate

# option B: create your own
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### Environment (`.env`)

Copy the repo-root `.env` (it already holds your API keys):

```bash
cp ../.env .env
```

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | **required** for the LLM judge (Google AI Studio) |
| `STT_MODEL_NAME` | `base` | faster-whisper model size |
| `EVAL_MODEL_NAME` | `gemini-3.5-flash` | LLM judge model |
| `EMBEDDINGS_ENABLED` | `true` | use the sentence-transformers semantic signal |
| `SCORE_THRESHOLD` | `90` | below this overall score → status `Mismatch` |

---

## Datasets (two, intentionally separate)

This project keeps **real evaluation data** and **synthetic test-case data** apart.

### Real data (what the application runs on)

```
datasets/recordings/                 actual audio (.mp4/.ogg)
datasets/manual_transcripts/         human gold references (.pdf)
datasets/stt_generated_transcripts/  Whisper output cached as (.txt)
```

These are the true inputs to `main.py` / `api.py`: the application transcribes a
real audio file into STT text and audits it against a real human transcript. This
is the actual demonstration of how the system works on real recordings — and is
**not** in JSON/CSV form. There are no expected categories attached to these
files because none of them have been manually labeled.

### Synthetic test cases (`dataset/test_cases.json`)

`dataset/test_cases.json` is the **submission's evaluation/test-case dataset**.
It is a direct export of the shared error-injection catalog in
`testing/scenarios.py` — the single source of truth that already parameterizes
the `pytest` suites of **both** approaches. Each case deliberately injects a
known transcription failure, e.g. `20 mg` → `200 mg` (expected: `incorrect`):

| name | gold / baseline | candidate / error_side | expected_a1_category |
|---|---|---|---|
| perfect_match | 20 mg aspirin … | 20 mg aspirin … | — (none) |
| number_change | 20 mg aspirin … | 200 mg aspirin … | incorrect |
| negated_fact | Anuria requires dialysis … | … does not require … | conflict |
| missing_item | … requires dialysis twice a week … | … requires twice a week … | missing |
| technical_word_garbled | … cholecystectomy … | … colosyctomy … | incorrect |
| contraction_equivalent | He'd had … | He had … | — (none) |
| short_spelling_noise | … seattle … | … seattl … | — (none) |

These cases are **not** generated from — and do not replace — the real audio,
manual-transcript, or STT files. They exist to verify that the evaluator
*detects and classifies* known failure modes.

- **One source of truth:** the JSON is regenerated straight from
  `testing/scenarios.py`, so the submitted dataset can never drift from what
  `pytest` asserts.
- **Not production input:** `main.py` / `api.py` never read `test_cases.json`;
  it exists solely as the submission/evaluation test-case dataset.
- **No invented labels:** expected categories come only from the (hand-labeled)
  catalog; real recordings carry no expected categories.

Regenerate the JSON when the catalog changes:

```bash
python -m testing.export_test_cases
```

### Evaluating accuracy

The catalog deliberately covers the five failure categories:
**Correct, Missing, Extra, Incorrect, Conflicting**. Run both approaches over
it and report per-category + overall accuracy:

```bash
python -m testing.eval_accuracy            # both approaches
python -m testing.eval_accuracy --approach 1
```

What the numbers mean (`Approach 1 (heuristic)` is the offline engine with no
LLM; `full pipeline` is the same engine assuming the LLM judge returns the right
verdict — i.e. the pipeline's own mapping accuracy), plus those for Approach 2's
deterministic detector:

- **Approach 1 (heuristic):** correct on Correct / Incorrect / Conflicting, but
  deliberately misses Missing / Extra / garbled-technical-word — those need the
  LLM layer, so the offline heuristic alone scores ~62%.
- **Approach 1 (full pipeline):** all categories map correctly → 8/8 (100%)
  given a correct judge verdict.
- **Approach 2 (detector):** flags and names every case correctly → 8/8 (100%).

Real-LLM accuracy (the judge actually classifying each case) requires API keys:

```bash
python -m testing.eval_layer3
```

## Running

### 1. CLI — evaluate an audio file against a gold transcript

```bash
python -m approach_1.main evaluate path/to/audio.ogg --gold path/to/gold.txt
```

- Transcribes with faster-whisper (`STT_MODEL_NAME`).
- Caches the transcript to
  `datasets/stt_generated_transcripts/<stem>_stt.txt`; pass `--force-stt` to
  re-transcribe regardless.
- Prints the full JSON report to stdout.

### 2. CLI — compare two transcript files (no audio / STT)

```bash
python -m approach_1.main evaluate-text --gold gold.txt --candidate candidate.txt
```

### 3. CLI — interactive runner (`compare.py`)

Lets you pick from the bundled dataset by index, or run fully offline.

```bash
python -m approach_1.compare                 # defaults to audio-3 / transcript_3
python -m approach_1.compare --audio 3 --gold 3
python -m approach_1.compare --audio audio-5.mp4 --gold transcript_5.pdf
python -m approach_1.compare --offline      # skip the LLM judge (heuristics only)
python -m approach_1.compare --csv report.csv
```

### 4. Web UI / API

```bash
uvicorn approach_1.api:app --reload --port 8000
```

Open <http://localhost:8000/> in a browser: upload the audio + a `.pdf`/`.txt`
manual transcript and click **Run**.

API endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | — | the review UI (HTML) |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/evaluate` | `multipart`: `audio`, `manual_transcript`, optional `gold_transcript` | `EvaluationReportV2` |
| `POST` | `/evaluate-text` | `form`: `gold_transcript`, `candidate_transcript` | `EvaluationReportV2` |

---

## The report

`EvaluationReportV2` contains:

- **signals** — `wer`, `cer`, `coverage`, `hallucination_ratio`,
  `entity_recall`, `entity_precision`, `semantic_similarity`
- **score_breakdown** — semantic / entity / lexical and the `error_penalty`
- **overall_score** — 0–100, `status` → `Match` or `Mismatch`
- **categorized findings** — `missing_information`, `incorrect_information`,
  `conflicting_information`, `hallucinated_information`, each item with the
  gold/candidate text, severity (low/medium/high) and an explanation
- **meta** — `llm_calls`, `latency_ms`, `generated_at`

---

## Tests

```bash
cd approach_1
pytest            # or: python -m pytest
```

From the repo root you can also run the whole suite with `make tests`.
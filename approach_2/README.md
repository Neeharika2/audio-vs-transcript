# Approach 2 — two engines, no gold transcript

No reference transcript needed. Two independent speech-to-text engines
(**Whisper** and **Deepgram**) transcribe the same audio, their word streams
are aligned, and every segment gets an **agreement score** (how closely the two
engines match). Only segments where the engines *disagree* are sent to an
**audio-capable LLM** (Gemini) that actually listens to the clip and decides
what was really said.

## How it works

```
audio ─► transcribed by Whisper AND Deepgram
          │
          ▼
   align words → diff → agreement score per segment
          │
   agreement ≥ 98%        → auto-accept, done
    90–97%                → review if it touches a technical term
    < 90% or suspicious   → LLM listens to the audio clip and
                            says which engine was right (if any)
          │
          ▼
   report → web review UI (optional human spot-check)
```

## Setup

One-time install (creates `.venv`, checks `ffmpeg`, checks `.env`):

```bash
make setup
```

Required API keys in the repo-root `.env`:

```
DEEPGRAM_API_KEY=...   # needed for the Deepgram engine
GEMINI_API_KEY=...     # needed for the LLM judge stage
```

Whisper runs locally (no key needed).

## Run

One command for everything, then open the web UI:

```bash
make run
```

Or run the steps individually:

| Step | Command | What it does |
|---|---|---|
| 1. Transcribe | `make transcribe` | both engines transcribe every file in `dataset/audio/` |
| 2. Review | `make review` | align, compare, score, and sample (free) |
| 3. Judge | `make judge` | LLM verifies flagged segments (Gemini, costs money) |
| UI | `make ui` | open http://localhost:8000 |

Same thing from the command line (.venv active):

```bash
python -m approach_2.main transcribe audio-3.ogg   # one file
python -m approach_2.main review audio-1          # build the report
python -m approach_2.main judge audio-1           # + LLM verdicts
uvicorn approach_2.api:app --port 8000            # review UI
```

## Output

```
dataset/whisper/<name>.txt, .segments.json      engine A, per word + confidence
dataset/deepgram/<name>.txt, .segments.json    engine B
dataset/review/<name>/report.json              aligned diff + scores + tiers
dataset/review/<name>/judgments.json           LLM verdicts for flagged segments
```

Each LLM verdict says: what was actually spoken, which engine (if either) erred,
the error type (`missing`/`extra`/`incorrect`/`conflicting`/`accurate`), and a
severity.

## Configuration

Everything is set via the repo-root `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | Whisper model size |
| `DEEPGRAM_API_KEY` | — | Deepgram (engine B) API key |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model |
| `GEMINI_API_KEY` | — | LLM judge API key |
| `GEMINI_MODEL` | `gemini-3.5-flash` | judge model |
| `LLM_DISAGREE_THRESHOLD` | `0.9` | agreement below this ⇒ LLM call |
| `REVIEW_DISAGREE_THRESHOLD` | `0.9` | agreement below this ⇒ always reviewed |

## Code layout

```
approach_2/
├── main.py         CLI
├── api.py          FastAPI web server (review UI)
├── config.py       .env settings
├── dataset/        audio, engine transcripts, reports
└── src/
    ├── align.py      word-stream alignment between engines
    ├── compare.py    per-segment diff + agreement (`1 − WER`)
    ├── score.py      confidence + review tier
    ├── review.py     spot-check sampling + acceptance
    ├── judge.py      critical-signal detection + Gemini audio judge
    ├── pipeline.py   end-to-end report builder
    └── models.py     data models
```

## Tests

```bash
make tests
```
# AI Framework for Comparing Different Data Formats

## Running approach_1

### Setup

```bash
cd approach_1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` from the project root and set your API keys (e.g. `GEMINI_API_KEY`):

```bash
cp ../.env .env   # or create one in approach_1/
```

### Configuration (via `.env`)

| Variable | Default | Options |
|---|---|---|
| `STT_PROVIDER` | `whisper` | `whisper` (local faster-whisper), `mock` (offline) |
| `STT_MODEL_NAME` | `base` | any faster-whisper model size |
| `EVAL_PROVIDER` | `gemini` | `gemini` (Gemini API), `mock` (offline heuristics) |
| `EVAL_MODEL_NAME` | `gemini-2.5-flash` | any Gemini model |
| `EMBEDDINGS_ENABLED` | `true` | `1`/`true`/`yes` to enable |
| `SCORE_THRESHOLD` | `90` | score below this -> status `Mismatch` |

### CLI

Evaluate audio against a gold transcript:

```bash
python -m approach_1.main evaluate audio.wav --gold gold.txt
```

Compare two transcripts directly (no audio/STT):

```bash
python -m approach_1.main evaluate-text --gold gold.txt --candidate candidate.txt
```

Run the synthetic validation suite:

```bash
python -m approach_1.main validate
```

### API server

```bash
uvicorn approach_1.api:app --reload
```

Endpoints:

- `GET /health` — health check
- `POST /evaluate` — form fields: `audio` (file) + `gold_transcript` (text)
- `POST /evaluate-text` — form fields: `gold_transcript` + `candidate_transcript`

### Tests

```bash
cd approach_1 && pytest
```

## Overview
This project compares two versions of the same information and automatically detects inconsistencies using an LLM. It identifies missing, incorrect, conflicting, and hallucinated information, then generates a structured evaluation report.

## Use Case
**Audio Transcript vs Summary**

## Features
- Detects missing information
- Identifies incorrect values
- Finds conflicting information
- Detects hallucinated (extra) information
- Generates structured evaluation reports

## Test Cases
The project includes:
- Perfect matches
- Missing information
- Incorrect values
- Ambiguous cases
- Sensitive information

## Sample Output
```json
{
  "missing_information": [],
  "incorrect_information": [],
  "conflicting_information": [],
  "hallucinated_information": [],
  "overall_score": 98,
  "status": "Match"
}
```



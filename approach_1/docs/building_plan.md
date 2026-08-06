# Phase-Wise Building Plan: Approach 1 (Gold Standard Evaluation)

This document maps out the implementation strategy for **Approach 1: Gold Standard Evaluation Framework**. 

---

# August 4, 2026
Target for today is to get the **end-to-end core pipeline** working with a **single example**. Do not build the whole project today (no dashboards, unit tests, or CLI suites). Focus strictly on getting one audio file transcribed and evaluated against a reference transcript, exposed through a simple FastAPI endpoint.

### Task 1: Finalize the Architecture
Document the transcription & evaluation pipeline:
```text
Audio ──► [ STT Model ] ──► Candidate Transcript
                                 │
                                 ├──► [ LLM Evaluator ] ──► JSON Report ──► Score
Gold Transcript ─────────────────┘
```
By the end of this step, I will have a clear layout of components and data flow.

### Task 2: Choose the STT Model
- **Decoupled Strategy**: Define an abstract `STTModel` protocol/interface.
- **Providers**: Implement `WhisperSTT` (local faster-whisper inference) as the primary provider, and `MockSTT` (offline mockup) for testing without audio files.
- Determine dynamic model loading based on a single config environment flag.

### Task 3: Create the Project Skeleton
Create the directory structure containing only the essential skeleton files:
```text
approach_1/
├── main.py          # Simple script to run end-to-end demo
├── config.py        # Settings (API keys) and provider factories
├── api.py           # FastAPI endpoint exposing /evaluate
├── requirements.txt # Essential dependencies
└── src/
    ├── models.py    # Pydantic schemas (ErrorItem, EvaluationReport)
    ├── evaluator.py # STT + LLM providers and the comparison prompt
    └── metrics.py   # Basic score aggregation placeholders
```
Do not build the dashboard, unit tests, or CLI suite yet.

### Task 4: Get STT Working
- Select **one** sample audio file.
- Implement Whisper STT execution to successfully transcribe the audio and verify the raw text output.

### Task 5: Build the Evaluation Prompt
- Draft the instructions for Gemini/LLM to audit and compare two transcripts (Gold vs Candidate).
- The prompt must request categorizing discrepancies into:
  - Missing information
  - Incorrect information
  - Conflicting information
  - Hallucinated information
- Enforce valid JSON structure.

### Task 6: Define the JSON Schema
- Write the initial Pydantic models for `ErrorItem` and `EvaluationReport` in `src/models.py`.

## Result of Step 1: End-to-End Core Pipeline Working

The core pipeline is functional: audio is transcribed with local Whisper (`faster-whisper`) and the candidate transcript is evaluated against a gold transcript by Gemini. Verified with a real audio file via the FastAPI `/evaluate` endpoint.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@harvard.wav;type=audio/wav' \
  -F 'gold_transcript=The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A hot cross bun.'
```

Response (`200 OK`):

```json
{
  "missing_information": [],
  "incorrect_information": [],
  "conflicting_information": [],
  "hallucinated_information": [
    {
      "category": "Extraneous detail",
      "reference_text": "A hot cross bun.",
      "generated_text": "A zestful food is the hot cross bun.",
      "context": null,
      "explanation": "The candidate transcript adds the descriptive phrase 'A zestful food is the' which is not present in the gold transcript.",
      "severity": "medium"
    }
  ],
  "overall_score": 90,
  "status": "Mismatch"
}
```

**Observations**
- The pipeline runs end-to-end on real audio and returns a structured report.
- The single flagged item is a Whisper transcription error ("A zestful food is the hot cross bun." vs gold "A hot cross bun."), which Gemini categorized as hallucinated information — a reminder that STT errors and true hallucinated content both surface here and may need to be distinguished later.
- `overall_score` and `status` come from Gemini, so they are model-dependent and may vary between runs.



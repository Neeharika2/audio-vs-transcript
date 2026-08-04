# Phase-Wise Building Plan: Approach 1 (Gold Standard Evaluation)

This document maps out the implementation strategy for **Approach 1: Gold Standard Evaluation Framework**. 

---

## August 4, 2026
Target for today is to get the **end-to-end core pipeline** working with a **single example**. Do not build the whole project today (no dashboards, APIs, CLI, or test runners). Focus strictly on getting one audio file transcribed and evaluated against a reference transcript.

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
- **Providers**: Implement `GeminiAudioSTT` (API cloud based) as primary, `LocalWhisperSTT` (local inference) as secondary, and `MockSTT` (offline mockup) for local testing without requirements.
- Determine dynamic model loading based on a single config environment flag.

### Task 3: Create the Project Skeleton
Create the directory structure containing only the essential skeleton files:
```text
approach_1/
├── main.py          # Temporary simple python script to run end-to-end demo
├── config.py        # Settings (API keys)
├── requirements.txt # Essential dependencies
├── dataset/         # Single test audio and gold transcript sample
└── src/
    ├── models.py    # Pydantic schemas (ErrorItem, EvaluationReport)
    ├── evaluator.py # LLM client & comparison prompt setup
    └── metrics.py   # Basic score aggregation placeholders
```
Do not build the FastAPI dashboard, unit tests, or CLI suite yet.

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

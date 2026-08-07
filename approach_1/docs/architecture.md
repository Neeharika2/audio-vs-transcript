# Architecture and Data Flow

This document details the architecture and data flow for **Approach 1: Gold Standard Evaluation Framework**. It describes how audio is transcribed and then evaluated against a human-verified ground truth transcript.

---

## 1. High-Level Architecture Diagram

The system operates in two distinct phases: **Transcription** (converting spoken audio to text) and **Evaluation** (comparing candidate transcript to the gold standard reference).

```text
               [ Raw Audio File (.wav/.mp3) ]
                             │
                             ▼
                    ┌─────────────────┐
                    │    STT Model    │  (Whisper Local / API)
                    └─────────────────┘
                             │
                             ▼
                 [ Generated Transcript ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Gold Transcript ]             [ Candidate Transcript ]
  (Human Ground Truth)             (System-Generated Text)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  LLM Evaluator  │  (Gemini API with Structured Prompt)
                    └─────────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │  Evaluation Report   │
                 │ ──────────────────── │
                 │ - Missing Info       │
                 │ - Incorrect Info     │
                 │ - Conflicting Info   │
                 │ - Hallucinated Info  │
                 │ - Consistency Score  │
                 └──────────────────────┘
```

---

## 2. Component Directory Structure

The skeletal project structure under `approach_1/` contains the following core modules:

*   **`main.py`**: The application driver that loads the audio and reference transcript, coordinates the STT runner and the LLM evaluator, and prints the final report.
*   **`config.py`**: Houses configuration variables, environment keys (like `DEEPSEEK_API_KEY`), and chosen models or weights.
*   **`requirements.txt`**: Declares third-party dependencies required for transcription and evaluation.
*   **`src/models.py`**: Contains Pydantic models enforcing strict JSON structures for evaluation outputs (`ErrorItem`, `EvaluationReport`).
*   **`src/evaluator.py`**: Manages LLM prompting, system instructions, and handles interactions with the DeepSeek API to retrieve structured auditing results.
*   **`src/metrics.py`**: Computes consistency metrics and calculates the overall matching score.

## 3. Modular Interface Decoupling

To ensure that the STT and LLM evaluation models can be changed easily without modifying core pipeline execution logic, both steps use abstract interface contracts.

### 3.1 Speech-to-Text (STT) Interface
All STT models implement a unified interface:
```python
class STTModel(Protocol):
    def transcribe(self, audio_path: str) -> str:
        ...
```
*   **LocalWhisperSTT**: Loads Whisper locally to transcribe via CPU/GPU.

The exact provider is selected in `config.py` via `get_stt_runner()` which returns the configured implementation.

### 3.2 LLM Evaluator Interface
All evaluation engines implement a unified interface:
```python
class LLMEvaluator(Protocol):
    def evaluate(self, gold_transcript: str, candidate_transcript: str) -> EvaluationReport:
        ...
```
*   **DeepSeekJudge**: Generates structured reports using DeepSeek Structured Outputs.

The engine is instantiated dynamically via `get_judge()` (model chosen by `EVAL_MODEL_NAME`). The driver (`main.py`) remains static and doesn't know which backend is executing the request.


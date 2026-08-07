# Evaluation Pipeline Plan: Ground Truth vs STT Transcript

## 1. Goal & Scope

Compare two text documents that represent the same spoken content:

- **Gold transcript** — human-verified ground truth (source of truth).
- **Candidate transcript** — STT model output (the artifact being evaluated).

The framework must detect and classify discrepancies into exactly four categories:

| Category | Meaning |
|---|---|
| `missing_information` | Fact present in gold, absent from candidate. |
| `incorrect_information` | Fact present in both, but stated differently in candidate (wrong value/wording that changes meaning). |
| `conflicting_information` | Candidate fact that directly contradicts gold. |
| `hallucinated_information` | Candidate content with no basis in gold (extra/made up). |

And produce a **structured evaluation report**: per-item findings with evidence spans, severity, plus an overall score and status.

### Current state (gap analysis)

`approach_1` already has a working skeleton: `WhisperSTT` transcribes audio, `GeminiEvaluator` sends one prompt with both transcripts and asks for a JSON report (`src/evaluator.py`, `src/models.py`). Weaknesses to fix in this plan:

1. **Single-pass, whole-document LLM call** — no normalization, no alignment; model decides everything, scores are model-dependent and unstable (noted in `building_plan.md`).
2. **No lexical/semantic signals** — WER/CER, embedding similarity, entity overlap are not computed.
3. **Categories are conflated** — e.g. Whisper mishearing is categorized as hallucination; there is no distinction between paraphrase, substitution, and hallucination.
4. **No validation of the evaluator itself** — no synthetic error-injection tests to measure detector precision/recall.
5. **No report artifacts** — only raw JSON; no per-segment diff view.

---

## 2. Why Naive String Comparison Fails

- Same meaning, different wording (STT paraphrases, filler words).
- STT punctuation/casing/formatting noise ("it's" vs "its", "50" vs "fifty").
- Word-order changes while meaning is preserved.
- Insertions/deletions shift positions, breaking positional diffing.

The pipeline therefore needs **layered signals** (lexical + semantic + entity) feeding a **small, focused LLM judgment per aligned segment**, not one giant comparison.

---

## 3. Proposed Pipeline

```text
Gold transcript ──┐
                 ├─► [1. Normalize] ─► [2. Segment & Align] ─► [3. Multi-signal extraction]
Candidate transcript ─┘                                            │
                                                                   ▼
                                                  [4. LLM classification per segment]
                                                                   │
                                                                   ▼
                                          [5. Aggregate & score] ─► [6. Report]
```

### Stage 1 — Normalization
Produce a canonical form for BOTH documents (used for lexical signals only; LLM sees the original text).

- Lowercase, strip punctuation, collapse whitespace.
- Number normalization: words ↔ digits (`fifty` → `50`).
- Common STT artifacts: expanded abbreviations, currency/units (`$50` ↔ `50 dollars`).
- Optional: speaker label/punctuation strip if present.

### Stage 2 — Segmentation & Alignment
Map candidate segments onto gold segments so each gold unit is checked exactly once.

- Split both docs into sentences (fallback: fixed-length windows for STT without punctuation).
- Align gold→candidate using a hybrid matcher:
  - Exact/char-level fuzzy match (rapidfuzz) as the anchor;
  - Sentence-embedding cosine similarity (**local `sentence-transformers`** — offline & deterministic) as the fallback for paraphrases;
  - Dynamic programming (Levenshtein-like sequence alignment) to handle insertions/deletions in the candidate.
- Output: aligned pairs `(gold_segment, candidate_segment, match_score)` plus **unmatched** gold segments and **unmatched** candidate segments.

Alignment is what makes the 4 categories *decidable*:
- Unmatched gold segment → `missing` (candidate) or `incorrect` if a weak match exists.
- Unmatched candidate segment → `hallucinated` (unless it's a weak paraphrase of a gold segment → `incorrect`).

### Stage 3 — Multi-Signal Extraction
For each aligned pair and for the full document, compute cheap deterministic signals (no LLM):

| Signal | Metric | Use |
|---|---|---|
| Lexical | WER, CER, character n-gram overlap | Raw transcription fidelity |
| Semantic | Sentence embedding cosine, paraphrase probability | Meaning preservation |
| Entity | NER recall/precision (names, dates, numbers, meds, places) | Domain-critical fidelity |
| Numeric | Exact numeric token match after normalization | Prices, dosages, dates |
| Length/coverage | Word count ratio, coverage of gold tokens | Missing/hallucination hint |

These signals do three jobs: (a) they surface as evidence in the report, (b) they help decide which pairs the LLM must examine in detail, and (c) they feed the final score so it is not LLM-pure.

### Stage 4 — LLM Classification (per segment)
Per-aligned-pair and per-unmatched-segment, call the LLM with a **small, targeted prompt**:

- Provide only the segment pair + surrounding context (2–3 segments) — not the whole document.
- Ask the model to classify the relationship: `match` | `missing` | `incorrect` | `conflict` | `hallucination`, with one short justification.
- Structured output enforced via Pydantic schema (reuse `ErrorItem` shape in `src/models.py`).
- Optional: a final global pass to catch cross-document contradictions that local passes miss (candidate says X, gold says not-X).

Rationale: splitting into segments makes the LLM more precise, cheaper per call, and gives us aligned evidence spans for the report. Whole-document single-call (current `approach_1`) remains as the fallback/backup.

### Stage 5 — Aggregation & Scoring
- Counts per category.
- **Severity-weighted score** computed from *deterministic* signals + LLM findings, e.g.:
  ```
  score = w1 * semantic_similarity
        + w2 * entity_recall
        + w3 * lexical_fidelity
        + w4 * (1 - weighted_error_penalty)
  ```
  weighted_error_penalty = Σ severity_weight × count per category.
  - `status = Match` iff no error with severity ≥ `high` **and** score ≥ 90 (threshold configurable). **Decision: any high-severity error OR score < 90 → Mismatch.**
- Keep Gemini-provided `overall_score` as an optional cross-check column, not the source of truth.

### Stage 6 — Report Generation
- Primary artifact: versioned JSON (`EvaluationReportV2`). **Decision: JSON only for now** — consumers build their own views.
- Each item must include `reference_text`, `generated_text`, `context`, `explanation`, `severity`, and `signal_evidence` (the deterministic metric values backing the claim).

---

## 4. Report Schema (V2)

```jsonc
{
  "id": "evl_8f2a",
  "inputs": {
    "gold_source": "harvard.wav.gold.txt",
    "candidate_source": "whisper-base.txt",
    "stt_model": "faster-whisper base",
    "evaluator": "deepseek-chat"
  },
  "alignment": {
    "gold_segments": 7,
    "candidate_segments": 7,
    "matched": 6,
    "unmatched_gold": 1,
    "unmatched_candidate": 1
  },
  "signals": {
    "wer": 0.06,
    "cer": 0.03,
    "semantic_similarity": 0.94,
    "entity_recall": 0.92,
    "entity_precision": 0.97,
    "coverage": 0.97
  },
  "findings": {
    "missing_information": [ { "reference_text": "...", "generated_text": null, "context": "...", "explanation": "...", "severity": "high", "signal_evidence": { "segment_score": 0.3 } } ],
    "incorrect_information": [],
    "conflicting_information": [],
    "hallucinated_information": []
  },
  "overall_score": 88,
  "status": "Mismatch",
  "score_breakdown": { "semantic": 94, "entity": 92, "lexical": 91, "error_penalty": -9 },
  "meta": { "llm_calls": 14, "latency_ms": 4100, "generated_at": "2026-08-05T..." }
}
```

---

## 5. Validation of the Evaluator (critical step)

An evaluation framework is only trustworthy if the evaluator itself is measured. Add a **synthetic test harness**:

1. **Error injection**: take a gold transcript, programmatically create N candidate variants, each with a known number of injected missing/incorrect/conflicting/hallucinated items (mutate words, drop clauses, swap numbers, insert sentences).
2. **Measure detector quality**: precision, recall, and F1 per category (do the detected items match the injected ground-truth annotations?).
3. **Human audit set**: small (10–20) real audio samples, human-annotated, for a spot check.
4. **Stability check**: run the evaluator 3× on identical input; the deterministic score must be identical and LLM findings should be ≥90% consistent.

This also gives a regression suite so prompt/model changes don't silently degrade detection.

---

## 6. Proposed Module Layout

```text
approach_1/
├── main.py               # CLI entry point (update)
├── config.py             # providers, thresholds, weights (update)
├── api.py                # FastAPI /evaluate (update schema)
├── requirements.txt
└── src/
    ├── models.py         # V2 schemas: Segment, AlignedPair, ErrorItemV2, EvaluationReportV2
    ├── normalize.py      # NEW: text normalization + number canonicalization
    ├── align.py          # NEW: segmentation + hybrid alignment (rapidfuzz + embeddings)
    ├── signals.py        # NEW: WER/CER/semantic/entity/numeric signals
    ├── classify.py       # NEW: per-segment + global LLM classification
    ├── score.py          # NEW: weighted aggregation (replaces metrics.py placeholder)
    ├── report.py         # NEW: JSON + Markdown/HTML report rendering
    ├── evaluate.py       # NEW: pipeline orchestrator (stages 1–6)
    ├── evaluator.py      # keep LLM providers, refactor to segment calls
    └── validation.py     # NEW: synthetic error injection + precision/recall harness
```

---

## 7. Phased Implementation Plan

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| **P0** — Framing | This plan | Approved scope & category definitions | **Done** |
| **P1** — Normalize | `src/normalize.py` + unit tests | `fifty dollars` ≡ `$50`; casing/punct noise removed | **Done** (25 tests) |
| **P2** — Align | `src/align.py`; segments & pairs | Unmatched gold/candidate segments correctly isolated | **Done** (13 tests) |
| **P3** — Signals | `src/signals.py`; WER/CER/semantic/entity/numeric | Deterministic metrics reproducible, unit-tested | **Done** (real local embeddings verified) |
| **P4** — Classify | `src/classify.py`; per-segment + global LLM with Pydantic output | 4 categories detected on synthetic examples | **Done** (offline heuristic fallback included) |
| **P5** — Score & Report | `src/score.py`, `src/evaluate.py`, V2 schemas | Full JSON report with score_breakdown + status | **Done** |
| **P6** — Validation | `src/validation.py`; error-injection harness | Precision/recall ≥0.85 per category on synthetic set | **Done** (1.0 offline on default suite) |
| **P7** — Integrate | `main.py`/`api.py`/`config.py`; GeminiJudge | `/evaluate` returns V2 report; CLI runs offline | **Done** |

**Verified:** 76 unit tests pass; `python -m approach_1.main validate` reports 1.0 precision/recall/F1 on the synthetic suite; FastAPI `/evaluate-text` returns the V2 report.

**Ordering rationale:** deterministic signals (P1–P3) come before LLM classification (P4) because the signals decide *where* the LLM looks and provide the score's backbone; the LLM is the classifier, not the source of truth.

---

## 8. Decisions (Locked)

1. **Judge model**: DeepSeek only — keep `deepseek-chat`; no OpenAI/multi-provider for now.
2. **Embeddings**: local `sentence-transformers` for the semantic signals (offline, deterministic).
3. **Category split**: *same fact restated with a different value → `incorrect`*; *candidate asserts the opposite of gold → `conflict`*.
4. **Report artifacts**: JSON only for v1.
5. **Status threshold**: `Mismatch` when any high-severity error OR `overall_score < 90`.

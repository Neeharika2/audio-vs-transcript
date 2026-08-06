# Implementation Plan: Consensus-Based STT Quality Assessment (Approach 2)

Implements the pipeline analyzed in `analysis.md`. The system transcribes one
audio file with **two independent STT engines**, aligns their segment streams,
diffs them word-by-word, scores each segment's confidence, and produces a
risk-tiered review checklist plus exports.

---

## 1. Goal & scope

**Goal:** Given an audio file, produce a per-segment confidence assessment of the
transcription, a prioritized review checklist (with the exact audio span for each
segment), and exportable artifacts (transcript + review report).

**Out of scope for v1:**
- Audio diarization / speaker labels.
- Storage backends (database). Review corrections persist to a JSON sidecar.
- A heavy web frontend. v1 UI is one static HTML page.
- DOCX/PDF export (added in a later phase; TXT/MD/SRT/VTT have no extra deps).

---

## 2. Pipeline

```text
[audio.wav]
   │
   ▼  src/audio.py     preprocess (16 kHz mono WAV, peak-normalize, optional noise reduction)
[audio_16k.wav]
   │
   ├──────────────┬──────────────────────────┐
   ▼              ▼                          ▼
src/engines.py   WhisperEngine           GoogleEngine (or MockEngine)
   │              │                        │
   │              ▼                        ▼
   │        segments A                  segments B
   │        (timestamps, word-level confidence)   (timestamps, word-level confidence)
   │              └──────────┬───────────┘
   ▼                         ▼
src/align.py     time-overlap + text-sim score matrix → Needleman–Wunsch → 1:N merge
   │                         │
   │              [AlignedSegment] pairs (combined time span)
   │                         ▼
src/compare.py   word-level edit diff per pair → per-word ops + agreement
   │                         ▼
src/score.py     per-segment confidence formula → review tier
   │                         ▼
src/review.py    spot-check sampling (tier0 + tier1 + seeded 10%) + acceptance
   │                         ▼
src/pipeline.py  assemble ReviewReport (JSON)   ──►  src/export.py → TXT/MD/SRT/VTT
   │
   ▼
api.py + static/review.html   interactive review (play span, diffs, inline corrections)
```

---

## 3. Module layout

```text
approach_2/
├── main.py               # CLI: preprocess | transcribe | review | export | validate
├── config.py             # providers, model sizes, weights, thresholds, seeds
├── api.py                # FastAPI: audio+segments endpoints, static review UI
├── requirements.txt
├── static/
│   └── review.html       # single-page review UI (phase P9)
├── docs/
│   ├── analysis.md
│   └── plan.md
└── src/
    ├── __init__.py
    ├── models.py         # Word, EngineSegment, AlignedSegment, WordOp, SegmentReview, ReviewReport
    ├── audio.py          # preprocess_audio() -> wav path
    ├── engines.py        # STTEngine protocol; WhisperEngine, GoogleEngine, MockEngine
    ├── normalize.py      # text normalization + filler stripping (reused patterns from approach_1)
    ├── align.py          # build_score_matrix(), needleman_wunsch(), merge_unmatched(), align()
    ├── compare.py        # word_diff(), agreement()
    ├── score.py          # segment_confidence(), assign_tier()
    ├── review.py         # sample_review_set(), acceptance_check()
    ├── pipeline.py       # run_pipeline() -> ReviewReport
    ├── export.py         # to_txt(), to_md(), to_srt(), to_vtt()
    └── validation.py     # synthetic two-engine fixtures + error injection
```

---

## 4. Data models (`src/models.py`, Pydantic)

```python
class Word(BaseModel):
    text: str
    start: float            # seconds
    end: float
    confidence: float | None  # engine-native, normalized to [0,1] by the engine

class EngineSegment(BaseModel):
    engine: str             # "whisper" | "google" | "mock"
    start: float
    end: float
    text: str
    confidence: float | None
    words: list[Word] = []

class AlignedSegment(BaseModel):
    idx: int
    start: float            # min(start) of constituent engine segments
    end: float              # max(end)
    engine_a: EngineSegment | None   # primary engine
    engine_b: EngineSegment | None   # secondary engine; None => no counterpart
    agreement: float | None          # 1 - WER over normalized words; None if one side missing
    diff: list[WordOp] = []

class WordOp(BaseModel):
    word: str
    op: str                 # "match" | "substitute" | "insert" | "delete"
    source: str             # "a" | "b"
    confidence_a: float | None
    confidence_b: float | None

class SegmentReview(BaseModel):
    segment: AlignedSegment
    confidence: float       # 0-100
    tier: str               # "auto_accept" | "review_technical" | "mandatory"
    verdict: str = "unreviewed"   # unreviewed | accepted | corrected | rejected
    correction: str | None = None # reviewer's inline fix

class SpotCheck(BaseModel):
    seed: int
    sample_ids: list[int]
    tiers: dict[str, list[int]]       # mandatory | disagreement | random_10pct
    accuracy: float | None
    accepted: bool | None
    expanded: bool = False

class ReviewReport(BaseModel):
    audio: str
    engines: list[str]
    segments: list[SegmentReview]
    spot_check: SpotCheck
    generated_at: str
```

---

## 5. Stage specs

### 5.1 Audio preprocessing (`audio.py`)

`preprocess_audio(src, out_dir, noise_reduction=False) -> Path`

- Convert to 16 kHz mono PCM WAV (via `pydub`/`ffmpeg`; fall back to `ffmpeg` CLI).
- Peak-normalize to −1 dBFS.
- If `noise_reduction=True`, apply a light noise-reduction pass (e.g. `noisereduce`).
  **Off by default.**
- Write to `out_dir/<name>_16k.wav`. Return the path. The API's review endpoints
  serve this normalized file so segment timestamps map directly to playback.

### 5.2 Engines (`engines.py`)

```python
class STTEngine(Protocol):
    def transcribe(self, audio_path: str) -> list[EngineSegment]: ...
```

- **WhisperEngine** — faster-whisper with `word_timestamps=True`,
  `vad_filter=True`. Map whisper confidence: `conf = 1 / (1 + exp(avg_logprob))`
  for segment level; word-level uses per-word logprob likewise. Calibration
  caveat in `analysis.md §4.2`.
- **GoogleEngine** — Google Cloud Speech with `enable_word_time_offsets=True`;
  per-word confidence is already `[0,1]`.
- **MockEngine** — reads a pre-exported JSON segment list (offline tests, no API).
- Factories in `config.py` (`get_engine_a()`, `get_engine_b()`), mirroring
  `approach_1/config.py`.

### 5.3 Normalization (`normalize.py`)

- Lowercase, strip punctuation, collapse whitespace, spell-out numbers/units
  (reuse the approach from `approach_1/src/normalize.py`).
- **Filler stripping:** remove a configurable filler set (`uh`, `um`, `uhh`,
  `mmm`, `hmm`, `er`, `ah`) **only for alignment/agreement** — the stored
  transcript keeps them. Track `stripped_fillers` count per segment.
- Used by `align.py` and `compare.py`; never mutates stored text.

### 5.4 Alignment (`align.py`) — the core

Inputs: `segments_a: list[EngineSegment]`, `segments_b: list[EngineSegment]`.

1. **Score matrix.** `score(i, j) = w_t·overlap(i, j) + w_s·text_sim(i, j)` where
   - `overlap = overlap_seconds / min(dur_i, dur_j)` (0 if disjoint);
   - `text_sim = token_set_ratio(norm_i, norm_j) / 100` (rapidfuzz);
   - default weights `w_t = 0.5`, `w_s = 0.5` (configurable).
2. **Needleman–Wunsch** over the matrix with a gap penalty and a match threshold
   (`MATCH_THRESHOLD = 0.55`). Produces the optimal alignment path → 1:1 pairs.
   Time is a strong prior but text similarity keeps matches correct when one
   engine's timestamps drift.
3. **1:N merge post-pass.** Any segment left unmatched on side B that is
   *adjacent* to a matched A-segment (gap ≤ 1 segment) is merged into that
   neighbor and re-tested against the A-segment (bounded to 3 segments per merge).
   Handles engines that split/merge sentences differently. Model this as
   `EngineSegment` concatenation with a merged time span.
4. **Unmatched remainder** becomes an `AlignedSegment` with the missing side
   `None` — by construction these are disagreements (tier 1, mandatory review).
5. **Combined span:** `start = min(a.start, b.start)`, `end = max(a.end, b.end)`.

Edge cases handled explicitly: empty transcription from one engine; silent
trailing regions; a constant timestamp offset (absorbed by the text-sim weight).

### 5.5 Word comparison (`compare.py`)

Per `AlignedSegment` (both sides present):

- Word-token Levenshtein alignment (`rapidfuzz.distance.Levenshtein` with edits
  ops, as in `approach_1/src/signals.py`) over **normalized** words → `WordOp`
  list.
- `agreement = 1 - distance / max(len_a, len_b)`.
- A word counts as low-confidence if its normalized confidence <
  `LOW_CONF_THRESHOLD = 0.6` on either side.
- If one side is missing, `agreement = 0.0`, `diff = []`, and the segment is a
  disagreement by definition.

### 5.6 Confidence scoring (`score.py`)

Per segment (both sides present):

```
engine_conf    = mean(conf of aligned words, averaged across both engines)
low_conf_ratio = (# words with conf < 0.6) / total words
agreement      = 1 - WER(norm_a, norm_b)

raw = 0.40·(100·engine_conf) + 0.45·(100·agreement) + 0.15·(100·(1 − low_conf_ratio))
raw -= 10.0 if (segment contains a glossary term AND agreement < 0.9)
confidence = clamp(round(raw), 0, 100)
```

When one side is missing, `confidence = 40 + 0.6·(100·engine_conf)` so unmatched
segments land in the mandatory tier but stay ordered by engine confidence.

**Tiers** (`assign_tier`):

| Confidence | Tier | Action |
|---|---|---|
| ≥ 98 | `auto_accept` | Accept automatically |
| 90–97 | `review_technical` | Accept unless glossary term present **or** agreement < 0.9 |
| < 90 | `mandatory` | Mandatory manual verification |

Weights, `LOW_CONF_THRESHOLD`, glossary path, and tier bounds all live in
`config.py`.

### 5.7 Spot-check sampling (`review.py`)

```python
def sample_review_set(report, seed, fraction=0.10):
    tier0 = [s for s in segments if s.tier == "mandatory"]
    tier1 = [s for s in segments if agreement is None or agreement < DISAGREE_THRESHOLD]
    tier2 = seeded_random_sample(
        [s for s in segments if s not in tier0 and s not in tier1], fraction
    )
    return tier0 + tier1 + tier2
```

- `DISAGREE_THRESHOLD = 0.9` (configurable).
- `fraction = 0.10` with a fixed `seed` (config) → reproducible samples.
- Sample order: mandatory → disagreement → random, so a reviewer processes
  highest-risk first.

**Acceptance** (`acceptance_check`):

- Reviewer marks each sampled word correct/incorrect against the audio
  (the UI plays the segment's exact time span).
- `accuracy = correct / reviewed_words`.
- `accuracy ≥ 0.99` → **accept** the transcript.
- `0.95 ≤ accuracy < 0.99` → **expand**: double the random sample, re-review,
  re-check.
- `< 0.95` → **full manual review** flag on the report.

### 5.8 Pipeline orchestration (`pipeline.py`)

`run_pipeline(audio_path, engines, seed, glossary) -> ReviewReport`

preprocess → transcribe A → transcribe B → align → compare → score → sample →
assemble `ReviewReport`. Deterministic given the same inputs and seed (whisper
inference is the only nondeterministic stage; document this).

### 5.9 Exports (`export.py`)

| Format | Contents | Dep |
|---|---|---|
| TXT | Transcript with per-segment confidence + tier | stdlib |
| Markdown | Transcript + diff table + tier badges | stdlib |
| SRT / VTT | Timestamped segments (best engine text, corrected text when available) | stdlib |
| DOCX / PDF | Phase 2 (needs `python-docx` / `reportlab`) | later |

### 5.10 Interactive review UI (Phase P9)

- FastAPI serves the normalized WAV (supports HTTP Range so `<audio>` can seek)
  and a `GET /segments` JSON (diff + tiers + spans).
- One static HTML page: segment list with confidence badge and tier color; click
  a segment → seek `<audio>` to `[start, end]` and play; highlight word ops
  (sub/ins/del colored); inline correction box saves to a JSON sidecar via
  `POST /review`.
- No build tooling, no JS framework.

---

## 6. Validation strategy

Mirror `approach_1/src/validation.py` (error-injection harness):

1. **Synthetic fixtures.** Build a gold segment stream with timestamps, then
   generate two independent engine outputs with *injected* per-engine errors:
   word substitutions, dropped segments, sentence merges/splits, filler words,
   and timestamp jitter.
2. **Alignment checks.** Assert the recovered pairs match the intended
   many-to-one mapping and that injected drops surface as missing-side segments.
3. **Diff checks.** Assert each injected substitution/insertion/delete is
   recovered as the right `WordOp`.
4. **Score checks.** Assert confidence and tier assignments fall where intended
   for clean, low-confidence, and disagreement segments.
5. **Spot-check determinism.** Same seed ⇒ same sample; acceptance gate behaves
   as specified for synthetic accuracies (1.0, 0.97, 0.90).
6. **Regression suite.** `python -m approach_2.main validate` prints precision /
   recall / F1 per check, matching `approach_1`'s exit-criteria style.

---

## 7. Phased implementation

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0** | `docs/analysis.md`, `docs/plan.md` | Approved scope; open questions answered |
| **P1** | Skeleton: `main.py`, `config.py`, `requirements.txt`, `src/models.py` | Models instantiate; CLI parses all subcommands |
| **P2** | `src/audio.py` | WAV → 16 kHz mono peak-normalized WAV; unit test on a generated tone |
| **P3** | `src/engines.py` (+ `MockEngine`) | Whisper + Mock transcribe the same file into `EngineSegment` lists |
| **P4** | `src/normalize.py`, `src/align.py` | Synthetic split/merge/drift cases align with correct 1:N pairs |
| **P5** | `src/compare.py` | Word ops and agreement correct on synthetic diffs |
| **P6** | `src/score.py` | Tier assignments match spec on clean/disagree/low-conf fixtures |
| **P7** | `src/review.py` | Seeded sampling + acceptance gate behave as specified |
| **P8** | `src/pipeline.py`, `src/export.py` | End-to-end `ReviewReport` + TXT/MD/SRT/VTT from a real audio file |
| **P9** | `api.py`, `static/review.html` | Click-to-play, diff highlights, inline corrections persist to JSON |
| **P10** | `src/validation.py` | Full suite passes; docs updated |

**Ordering rationale:** deterministic core (align → compare → score) before any
UI or API, exactly as in `approach_1`. Alignment (P4) is the riskiest stage and
gets the most test coverage.

---

## 8. Config surface (`config.py`)

| Key | Default | Meaning |
|---|---|---|
| `ENGINE_A` | `whisper` | `whisper` (faster-whisper) or `mock` |
| `WHISPER_MODEL` | `base` | faster-whisper size (`base` for tests, `large-v3` for prod) |
| `ENGINE_B` | `google` | `google` (Cloud Speech) or `mock` |
| `NOISE_REDUCTION` | `false` | enable preprocessing noise reduction |
| `ALIGN_WEIGHT_TIME` | `0.5` | time-overlap weight in alignment score |
| `ALIGN_WEIGHT_TEXT` | `0.5` | text-similarity weight in alignment score |
| `MATCH_THRESHOLD` | `0.55` | alignment match cutoff |
| `CONF_WEIGHTS` | `{engine: 0.40, agreement: 0.45, low_conf: 0.15}` | score weights |
| `LOW_CONF_THRESHOLD` | `0.6` | per-word confidence cutoff |
| `TIER_AUTO_ACCEPT` | `98` | ≥ this ⇒ auto-accept |
| `TIER_REVIEW` | `90` | below this ⇒ mandatory |
| `DISAGREE_THRESHOLD` | `0.9` | agreement below ⇒ disagreement tier |
| `SPOT_CHECK_FRACTION` | `0.10` | random sample fraction |
| `SPOT_CHECK_SEED` | `42` | sampling seed (reproducible) |
| `SPOT_CHECK_ACCEPT` | `0.99` | sample accuracy to accept |
| `GLOSSARY_PATH` | (empty) | optional domain term list |

---

## 9. Decisions (locked)

1. Consensus evaluation with two engines; no human reference transcript.
2. Alignment is time+text Needleman–Wunsch with a 1:N merge pass — never
   line-by-line comparison.
3. Agreement = `1 - WER`; confidence is the deterministic weighted formula in §5.6.
4. Review tiers and spot-check rules per the user's thresholds (§5.6, §5.7).
5. Engine 2 = Google Cloud Speech; `MockEngine` for offline tests.
6. Exports: TXT/MD/SRT/VTT in v1; DOCX/PDF in a later phase.
7. Review UI is a dependency-free static HTML page served by FastAPI.

## 10. Open questions (blocking P3/P9)

1. Confirm Google Cloud Speech is available (service-account credentials,
   `GOOGLE_APPLICATION_CREDENTIALS`), or switch secondary engine to Azure Speech.
2. Language: default English — confirm before the engines are wired.
3. Whisper model size for production runs (`base` vs `large-v3`): affects speed
   and quality trade-offs.
4. Glossary source: wordlist file acceptable, or is a richer entity list expected?

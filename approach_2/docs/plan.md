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
   ▼  src/audio.py     ffmpeg: 16 kHz mono WAV
[audio_16k.wav]
   │
   ├────────────────┬──────────────────┐
   ▼                ▼                  ▼
src/engines.py   WhisperEngine    DeepgramEngine
   │                │                │
   │                ▼                ▼
   │          segments A         segments B
   │          (timestamps, word confidence)
   │                └──────┬───────┘
   ▼                       ▼
src/align.py     global word-level Needleman–Wunsch → regroup into AlignedSegments
   │                       ▼
   │            [AlignedSegment] pairs (combined time span)
   │                       ▼
src/compare.py   word-level edit diff per pair → per-word ops + agreement
   │                       ▼
src/score.py     per-segment confidence → review tier
   │                       ▼
src/review.py    spot-check sampling (mandatory + disagreement + seeded 10%) + acceptance
   │                       ▼
src/pipeline.py  assemble ReviewReport   ──►  src/export.py → TXT/MD/SRT/VTT
   │                       │
   │        suspicious segments only (agreement < 0.9 or missing side)
   │                       ▼
src/judge.py     audio-grounded LLM (Gemini 3.5 Flash) → classify + severity
   │                       ▼
   │        llm_judgment attached to AlignedSegment → exports / API
   │
   ▼
api.py + static/review.html   interactive review (play span, diffs, inline corrections)
```

---

## 3. Module layout

```text
approach_2/
├── main.py               # CLI: preprocess | transcribe | review | export
├── config.py             # engine, thresholds, seed, glossary
├── api.py                # FastAPI: audio + segments endpoints, static review UI
├── requirements.txt
├── static/
│   └── review.html       # single-page review UI (phase P8)
├── docs/
│   ├── analysis.md
│   └── plan.md
├── src/
│   ├── __init__.py
│   ├── models.py         # Word, EngineSegment, WordOp, AlignedSegment, SpotCheck, ReviewReport
│   ├── audio.py          # preprocess_audio() -> wav path
│   ├── engines.py        # WhisperEngine, DeepgramEngine
│   ├── align.py          # align() (global word-level NW + window reassembly)
│   ├── compare.py        # word_diff(), agreement()
│   ├── score.py          # segment_confidence(), assign_tier()
│   ├── review.py         # sample_review_set(), acceptance_check()
│   ├── judge.py          # select_for_judgment(), LLMJudge/GeminiJudge, judge_report()
│   ├── pipeline.py       # run_pipeline() -> ReviewReport
│   └── export.py         # to_txt(), to_md(), to_srt(), to_vtt()
└── tests/
    ├── test_align.py     # synthetic split/merge/drift fixtures
    ├── test_compare.py
    ├── test_score.py
    └── test_review.py
```

Notes:
- **`normalize.py`** — local copy at `approach_2/src/normalize.py` (kept
  separate from approach_1 per the repo convention). It imports only stdlib.
  Filler stripping is a tiny helper in `align.py`.
- **No engine protocol, no mock engine** — the two concrete engines are the only
  ones; tests build `EngineSegment` objects directly. This matches the direction
  of the recent `approach_1` refactor.
- **No `validation.py`** — synthetic fixtures and assertions live in `tests/`
  and run via pytest.

---

## 4. Data models (`src/models.py`, Pydantic)

```python
class Word(BaseModel):
    text: str
    confidence: float | None   # engine-native, normalized to [0,1] by the engine

class EngineSegment(BaseModel):
    engine: str             # "whisper" | "google"
    start: float            # seconds
    end: float
    text: str
    confidence: float | None
    words: list[Word] = []

class WordOp(BaseModel):
    text: str
    op: str                 # "match" | "substitute" | "insert" | "delete"

class AlignedSegment(BaseModel):
    idx: int
    start: float            # min(start) of constituent engine segments
    end: float              # max(end)
    engine_a: EngineSegment | None   # primary engine
    engine_b: EngineSegment | None   # secondary engine; None => no counterpart
    agreement: float | None          # 1 - WER over normalized words; None if one side missing
    diff: list[WordOp] = []
    confidence: float = 0.0          # 0-100 (filled by score.py)
    tier: str = "mandatory"          # "auto_accept" | "review_technical" | "mandatory"
    verdict: str = "unreviewed"      # unreviewed | accepted | corrected | rejected
    correction: str | None = None    # reviewer's inline fix

class SpotCheck(BaseModel):
    seed: int
    sample_ids: list[int]
    accuracy: float | None
    accepted: bool | None
    expanded: bool = False

class ReviewReport(BaseModel):
    audio: str
    engines: list[str]
    segments: list[AlignedSegment]
    spot_check: SpotCheck
    generated_at: str
```

---

## 5. Stage specs

### 5.1 Audio preprocessing (`audio.py`)

`to_wav_16k(src, out_dir) -> Path`

- Shell out to `ffmpeg`: resample to 16 kHz mono PCM WAV (`pcm_s16le`). One
  command, no new dependencies. No loudness normalization or noise reduction
  (kept deliberately conservative — both engines work fine on the raw signal).
- Write to `out_dir/<name>.wav`. The API's review endpoints serve this file
  so segment timestamps map directly to playback.

### 5.2 Engines (`engines.py`)

Two concrete classes; no shared protocol.

- **WhisperEngine** — faster-whisper with `word_timestamps=True`,
  `vad_filter=True`. Segment confidence is `exp(avg_logprob)`; word confidence
  is the per-token probability.
- **DeepgramEngine** — Deepgram Nova over REST (`POST /v1/listen`) with
  `utterances=true` + `words=true`; per-word and utterance confidence come back
  in `[0,1]`. Smart formatting / punctuation are disabled so the transcript
  comes back as plain words, matching Whisper's output for alignment. Requires
  `DEEPGRAM_API_KEY`.
- Both return `list[EngineSegment]`; `main.py` instantiates one of each.

### 5.3 Alignment (`align.py`) — the core

Inputs: `segments_a`, `segments_b` (both `list[EngineSegment]`).

1. **Normalize each segment text** with `approach_2.src.normalize.normalize_text`
   and strip the shared filler set (`uh`, `um`, `uhh`, `mmm`, `hmm`, `er`, `ah`)
   — only for matching; stored text is never mutated.
2. **Flatten to word streams.** Every engine-A and engine-B word becomes a token
   with an interpolated timestamp (each word is placed evenly across its
   segment's span).
3. **Global Needleman–Wunsch over word tokens.** Equal words score +1,
   substitutions and gaps score −1; a diagonal match is additionally penalized
   when its two interpolated timestamps differ by more than `_TIME_TOLERANCE`
   (`1.5 s`) at `_TIME_PENALTY` (`2.0`) per extra second. This keeps a repeated
   phrase from being smeared to a different spoken span when one engine drops a
   segment. Every non-gap cell (match or substitution) is kept as a
   correspondence.
4. **Window reassembly.** Open one window per engine-A segment and merge
   adjacent windows whenever a single engine-B segment contributes words to both
   — engine B spans the boundary, so the split is only engine A's. This absorbs
   1:N, N:1, and crossing (2×2) splits without choosing either engine as a
   reference.
5. **Assign engine-B segments whole** to the window holding their matched words.
   Segments with no matched words go to the best time-overlapping window, or
   become engine-B-only segments when they cover audio engine A never
   transcribed (genuine unmatched content — a disagreement by definition).
6. **Combined span:** `start = min(a.start, b.start)`, `end = max(a.end, b.end)`.

Word-level comparison happens later in `compare()`, per window, on these
already-aligned spoken spans.

Edge cases handled explicitly: empty transcription from one engine; silent
trailing regions; timestamp drift (covered by the time penalty + word order).

### 5.4 Word comparison (`compare.py`)

Per `AlignedSegment` with both sides present:

- Levenshtein alignment over **normalized** word tokens (hand-rolled DP in
  `compare.py`, stdlib only) → `WordOp` list.
- `agreement = 1 - distance / max(len_a, len_b)`.
- Missing side ⇒ `agreement = 0.0`, `diff = []`.

### 5.5 Confidence scoring (`score.py`)

One formula for every segment (a missing side simply has `agreement = 0`):

```
engine_conf    = mean word confidence across both engines
low_conf_ratio = count(words with conf < 0.6) / total words
agreement      = 1 - WER(norm_a, norm_b)

confidence = round(100 · (0.40·engine_conf + 0.45·agreement + 0.15·(1 − low_conf_ratio)))
confidence = clamp(confidence, 0, 100)
```

Weights are module constants, not config. A missing-side segment lands at
~40–55 → mandatory tier, which is correct by construction.

**Tiers** (`assign_tier`) — exactly the user's rule:

| Confidence | Tier | Action |
|---|---|---|
| ≥ 98 | `auto_accept` | Accept automatically |
| 90–97 | `review_technical` | Review only if the segment contains a glossary term |
| < 90 | `mandatory` | Mandatory manual verification |

`LOW_CONF_THRESHOLD`, tier bounds, and glossary path live in `config.py`.
Disagreement segments (agreement < 0.9) are always reviewed regardless of tier —
handled by spot-check sampling (§5.6), not by the tier rule.

### 5.6 Spot-check sampling (`review.py`)

```python
def sample_review_set(report, seed, fraction=0.10):
    mandatory = [s for s in report.segments if s.tier == "mandatory"]
    disagree  = [s for s in report.segments
                 if s.agreement is None or s.agreement < DISAGREE_THRESHOLD]
    rest      = [s for s in report.segments
                 if s not in mandatory and s not in disagree]
    random_10 = seeded_random_sample(rest, fraction)   # seed from config
    return mandatory + disagree + random_10
```

- Review order is highest-risk first (mandatory → disagreement → random 10%).
- Fixed seed in config ⇒ reproducible samples.

**Acceptance** (`acceptance_check`):

- Reviewer marks each sampled segment correct/incorrect against the audio
  (the UI plays the segment's exact time span).
- `accuracy ≥ 0.99` → **accept** the transcript.
- `0.95 ≤ accuracy < 0.99` → **expand**: review more segments.
- `< 0.95` → **full manual review** flag on the report.
- **Status:** implemented as `apply_review()` in `review.py`; verdicts persist
  to a JSON sidecar and are merged on load by both the CLI (`review`) and the API.

### 5.7 Pipeline orchestration (`pipeline.py`)

`run_pipeline(audio_path, engines, seed, glossary) -> ReviewReport`

preprocess → transcribe A → transcribe B → align → compare → score → sample →
assemble `ReviewReport`. Deterministic given the same inputs and seed (whisper
inference is the only nondeterministic stage; document this).

### 5.8 Exports (`export.py`)

| Format | Contents | Dep |
|---|---|---|
| TXT | Transcript with per-segment confidence + tier | stdlib |
| Markdown | Transcript + diff table + tier badges | stdlib |
| SRT / VTT | Timestamped segments (best engine text, corrected text when available) | stdlib |
| DOCX / PDF | Phase 2 (needs `python-docx` / `reportlab`) | later |

### 5.9 Interactive review UI (implemented)

- FastAPI (`api.py`) serves the source audio (HTTP Range so `<audio>` can seek),
  a `GET /report/{stem}` JSON (diff + tiers + spans), and accepts verdicts via
  `POST /review/{stem}`.
- One static HTML page (`static/review.html`): segment list with confidence
  badge and tier color; click a segment → seek `<audio>` to `[start, end]` and
  play; highlight word ops (sub/ins/del colored); mark correct/incorrect and
  save corrections.
- Verdicts persist to `dataset/review/<name>/verdicts.json`.
- No build tooling, no JS framework.

### 5.10 Audio-grounded LLM judge (`judge.py`)

The two STT engines are a cheap first-pass filter. Only segments the
deterministic disagreement detector flags are sent to an audio-capable LLM
(Gemini 3.5 Flash) that listens to the actual audio span and arbitrates.

- **Detector** (`select_for_judgment`): a segment is suspicious iff one side is
  missing (`engine_a`/`engine_b` is `None`) or `agreement < DISAGREE_THRESHOLD`.
  Everything else is high-confidence and bypasses the LLM entirely.
- **LLM input** per flagged segment: the audio span cut with ffmpeg
  (`audio.extract_span`, padded by `JUDGE_PAD_SECONDS`), both transcripts, the
  aligned timestamps, and per-engine confidence.
- **Structured output** (`LLMJudgeVerdict`): `classification`
  (`missing | extra | hallucinated | incorrect | conflicting | accurate`),
  `correct_content`, `whisper_error`, `deepgram_error`, `severity`
  (`low | medium | high | critical`), `explanation`, `evidence`.
- **Modularity**: `LLMJudge` is a protocol; `GeminiJudge` is the implementation
  (google-genai SDK, lazily imported so tests never need it). Swap = new class.
- Verdicts persist to `dataset/review/<name>/judgments.json` and are merged on
  load by the API; exports prefer `correct_content` when present.
- CLI: `python -m approach_2.main judge [audio]` (needs `GEMINI_API_KEY`).

---

## 6. Testing strategy

Plain pytest tests, written alongside each stage, using synthetic fixtures:

1. **Alignment fixtures.** A gold segment stream with timestamps; two engine
   outputs with injected substitutions, dropped segments, sentence
   merges/splits, filler words, and timestamp jitter.
2. **Alignment tests** — recovered pairs match the intended 1:N mapping;
   injected drops surface as missing-side segments.
3. **Compare tests** — each injected substitution/insert/delete is recovered as
   the right `WordOp`; agreement is correct.
4. **Score tests** — confidence and tier assignments for clean, low-confidence,
   and missing-side segments.
5. **Review tests** — same seed ⇒ same sample; acceptance gate behaves for
   synthetic accuracies (1.0, 0.97, 0.90).
6. **End-to-end** — `run_pipeline` on a short real audio file (if a second
   engine is available) or on synthetic segment lists, asserting a valid
   `ReviewReport` and exports.

---

## 7. Config surface (`config.py`)

Every key maps to a stated requirement or an explicit user threshold:

| Key | Default | Meaning |
|---|---|---|
| `ENGINE_A` | `whisper` | primary engine |
| `WHISPER_MODEL` | `small` | faster-whisper size (`base` for tests, `small`/`large-v3` for prod) |
| `ENGINE_B` | `deepgram` | secondary engine |
| `DEEPGRAM_API_KEY` | (env) | Deepgram Nova API key |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model name |
| `LOW_CONF_THRESHOLD` | `0.6` | per-word confidence cutoff |
| `TIER_AUTO_ACCEPT` | `98` | ≥ this ⇒ auto-accept |
| `TIER_REVIEW` | `90` | below this ⇒ mandatory |
| `DISAGREE_THRESHOLD` | `0.9` | agreement below ⇒ always reviewed |
| `SPOT_CHECK_FRACTION` | `0.10` | random sample fraction |
| `SPOT_CHECK_SEED` | `42` | sampling seed (reproducible) |
| `SPOT_CHECK_ACCEPT` | `0.99` | sample accuracy to accept |
| `GLOSSARY_PATH` | (empty) | optional domain term list |
| `GEMINI_API_KEY` | (env) | Google AI Studio key for the LLM judge |
| `GEMINI_MODEL` | `gemini-3.5-flash` | audio-capable judge model |
| `JUDGE_PAD_SECONDS` | `0.5` | padding on each side of a cut audio span |

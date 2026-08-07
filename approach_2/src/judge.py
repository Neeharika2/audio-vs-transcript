"""Audio-grounded LLM judge for disagreement segments.

The two STT engines are the cheap first-pass filter. Only segments the
deterministic detector flags are sent to an audio-capable LLM. A segment is
flagged when it has a **critical signal** (negation, number, glossary term, or
content-word change) regardless of its agreement score, or when the engines
agree below `LLM_DISAGREE_THRESHOLD` / one side is missing. The LLM listens to
the actual audio clip and arbitrates between the two transcripts, returning a
structured verdict (classification, severity, per-engine error flags, correct
content).

`LLMJudge` is the interface; `GeminiJudge` is the Gemini 3.5 Flash
implementation. The SDK is imported lazily so the rest of the pipeline (and the
test suite) works without it installed or without a key.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Protocol

from approach_2 import config
from approach_2.src.align import norm_words
from approach_2.src.audio import extract_span
from approach_2.src.models import AlignedSegment, LLMJudgeVerdict, ReviewReport
from approach_2.src.review import load_glossary


# ---------------------------------------------------------------------------
# Critical-signal detection (deterministic; no LLM involved)
#
# Agreement is a *lexical* similarity score: "the patient" vs "patient" scores
# nearly 1.0 even though they mean the same thing, while "requires" vs "does
# not require" is a semantic contradiction that a single missing word can mask.
# A threshold alone cannot separate the two, so specific high-risk changes are
# escalated to the LLM regardless of the overall agreement score. These rules
# are intentionally conservative (a few cheap extra LLM calls) because missing
# a medication/negation change is worse than over-checking a harmless plural.
# ---------------------------------------------------------------------------

# Normalized tokens that flip meaning when they appear on one side only.
_NEGATION_TOKENS = {
    "not", "no", "never", "none", "without", "doesnt", "dont", "didnt",
    "cannot", "cant", "wont", "isnt", "arent", "nt", "unlikely",
}

# Filler + function words that carry no clinical meaning; a change in these is
# not a critical signal.
_TRIVIAL_TOKENS = {
    "uh", "um", "uhh", "umm", "hmm", "mmm", "hm", "mm", "er", "ah", "oh",
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "is", "was", "are", "were", "has", "have", "had", "it", "its",
}

# A substituted word on either side at least this long is treated as a
# technical/domain term worth checking ("cholecystectomy" vs "colosyctomy"),
# while short substitutions are usually spelling or plural noise.
_TECHNICAL_LENGTH = 8


def _token_pairs(seg: AlignedSegment) -> list[tuple[str | None, str | None]]:
    """Align the two engines' normalized tokens; yield (a_token, b_token) pairs.

    A `None` side means an insertion (b-only) or deletion (a-only). This is a
    straight word-level Levenshtein backtrack (word order is identical across
    engines), so substitutions are distinguished from insertions/deletions.
    """
    a_tokens = norm_words(seg.engine_a) if seg.engine_a else []
    b_tokens = norm_words(seg.engine_b) if seg.engine_b else []
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_tokens[i - 1] == b_tokens[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            pairs.append((a_tokens[i - 1], b_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((a_tokens[i - 1], b_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((a_tokens[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b_tokens[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def critical_difference(seg: AlignedSegment) -> str | None:
    """Return a reason string if the segment carries a critical signal, else None.

    Critical signals (any one is enough to send the segment to the LLM):
      - negation changed ("requires" -> "does not require")
      - a number differs ("20 mg" vs "200 mg", "October 6" vs "sixth")
      - a glossary/domain term changed (medication, anatomy, procedure)
      - a content word was inserted or deleted (meaning added or lost)
      - a long technical word was substituted (likely a domain term garbled)

    Short word substitutions ("seattl"/"seattle", "spray"/"sprays") are treated
    as spelling/plural noise and left to the agreement threshold.
    """
    if seg.engine_a is None or seg.engine_b is None:
        return "missing_side"

    pairs = _token_pairs(seg)
    glossary = load_glossary()

    # Strong signals first, segment-wide: a negation or number anywhere on one
    # side is critical regardless of which aligned pair it lands in.
    all_tokens = set()
    for a_tok, b_tok in pairs:
        if a_tok == b_tok:
            continue
        all_tokens |= {t for t in (a_tok, b_tok) if t}
    if all_tokens & _NEGATION_TOKENS:
        return "negation"
    if any(t.isdigit() for t in all_tokens):
        return "number"
    if glossary & all_tokens:
        return "glossary_term"

    for a_tok, b_tok in pairs:
        if a_tok == b_tok:
            continue
        both = {t for t in (a_tok, b_tok) if t}

        # True insertion/deletion of a content word changes meaning.
        if (a_tok is None or b_tok is None) and any(
            t not in _TRIVIAL_TOKENS and len(t) > 2 for t in both
        ):
            return "word_added_removed"

        # Long substituted word = likely a garbled technical/domain term.
        if both and any(len(t) >= _TECHNICAL_LENGTH for t in both):
            return "technical_word"

    return None


# ---------------------------------------------------------------------------
# Disagreement detector (deterministic; no LLM involved)
# ---------------------------------------------------------------------------

def select_for_judgment(
    report: ReviewReport,
    disagree_threshold: float | None = None,
) -> list[AlignedSegment]:
    """Flag segments that need audio-grounded LLM verification.

    A segment is flagged iff it carries a critical signal OR (one side is
    missing or agreement < `disagree_threshold`, default
    `config.LLM_DISAGREE_THRESHOLD`). Everything else bypasses the LLM.
    """
    threshold = config.LLM_DISAGREE_THRESHOLD if disagree_threshold is None else disagree_threshold
    return [
        s for s in report.segments
        if critical_difference(s) is not None
        or s.agreement is None
        or s.agreement < threshold
    ]


def is_suspicious(segment: AlignedSegment, disagree_threshold: float | None = None) -> bool:
    threshold = config.LLM_DISAGREE_THRESHOLD if disagree_threshold is None else disagree_threshold
    return (
        critical_difference(segment) is not None
        or segment.agreement is None
        or segment.agreement < threshold
    )


# ---------------------------------------------------------------------------
# Judge interface + request
# ---------------------------------------------------------------------------

class JudgeRequest:
    """Everything the LLM needs for one disagreement segment."""

    def __init__(
        self,
        segment: AlignedSegment,
        audio_bytes: bytes,
        mime_type: str = "audio/wav",
    ):
        self.segment = segment
        self.audio_bytes = audio_bytes
        self.mime_type = mime_type


class LLMJudge(Protocol):
    """Swappable audio-capable judge. Returns a verdict or None on failure."""

    def judge(self, request: JudgeRequest) -> LLMJudgeVerdict | None:
        ...


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------

_CLASSIFICATION_HINTS = """\
missing        — one transcript omitted content that IS in the audio
extra          — a transcript contains content NOT in the audio (words added)
hallucinated   — a transcript invented words/phrases that are not in the audio
incorrect      — a word or short span is transcribed wrongly in one/both engines
conflicting    — the two transcripts conflict on something material (e.g. a dose)
accurate       — after listening, the audio matches (a false alarm); no error
"""

_PROMPT_TEMPLATE = """\
You are verifying medical transcription. Two independent speech-to-text engines
(Whisper and Deepgram) transcribed the SAME audio clip. They disagree on the
span below. Neither engine is ground truth.

Your job: LISTEN to the attached audio and determine what was actually spoken.
Decide which engine (if either) is correct, or whether both are wrong.

Audio clip: {start:.2f}s - {end:.2f}s

Whisper (confidence {whisper_conf}):
"{whisper_text}"

Deepgram (confidence {deepgram_conf}):
"{deepgram_text}"

Classify the disagreement as one of:
{_CLASSIFICATION_HINTS}
Reply ONLY with a JSON object, no prose, in this exact shape:
{{
  "classification": "missing|extra|hallucinated|incorrect|conflicting|accurate",
  "correct_content": "what was actually spoken, verbatim, or \\"\\" if unknown",
  "whisper_error": true|false,
  "deepgram_error": true|false,
  "severity": "low|medium|high|critical",
  "explanation": "short plain-English reason, grounded in what you heard",
  "evidence": "the exact words you heard in the audio that justify the verdict"
}}
"""


def build_prompt(segment: AlignedSegment) -> str:
    """Compose the judge prompt from an aligned segment (transcripts + timestamps + confidence)."""
    a, b = segment.engine_a, segment.engine_b
    whisper_text = a.text if a else "(engine produced no text for this span)"
    deepgram_text = b.text if b else "(engine produced no text for this span)"
    whisper_conf = f"{a.confidence:.2f}" if a and a.confidence is not None else "n/a"
    deepgram_conf = f"{b.confidence:.2f}" if b and b.confidence is not None else "n/a"
    return _PROMPT_TEMPLATE.format(
        start=segment.start,
        end=segment.end,
        whisper_text=whisper_text,
        deepgram_text=deepgram_text,
        whisper_conf=whisper_conf,
        deepgram_conf=deepgram_conf,
        _CLASSIFICATION_HINTS=_CLASSIFICATION_HINTS,
    )


def parse_verdict(text: str) -> LLMJudgeVerdict | None:
    """Extract a structured verdict from the model's text reply.

    The model is told to reply with bare JSON; tolerate ```json fences, leading
    prose, and trailing text by locating the first `{` and last `}`.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    try:
        return LLMJudgeVerdict.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------

class GeminiJudge:
    """Audio-grounded judge backed by Gemini 3.5 Flash (google-genai SDK)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self.model = model if model is not None else config.GEMINI_MODEL
        if not self.api_key:
            raise ValueError(
                "GeminiJudge requires an API key; set GEMINI_API_KEY in the repo-root .env"
            )
        self._client = None
        self._MAX_RETRIES = 3
        self._RETRY_DELAY = 2.0

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def judge(self, request: JudgeRequest) -> LLMJudgeVerdict | None:
        from google.genai import types
        client = self._get_client()
        prompt = build_prompt(request.segment)
        last_error: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=request.audio_bytes, mime_type=request.mime_type),
                        prompt,
                    ],
                )
                return parse_verdict(response.text or "")
            except Exception as exc:  # transient API/rate errors; retry, then skip
                import sys
                print(
                    f"  judge attempt {attempt + 1}/{self._MAX_RETRIES} failed on segment "
                    f"{request.segment.idx}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                last_error = exc
                if attempt < self._MAX_RETRIES - 1:
                    import time
                    time.sleep(self._RETRY_DELAY * (attempt + 1))
        print(f"  judge gave up on segment {request.segment.idx} after {self._MAX_RETRIES} attempts", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def judge_report(
    report: ReviewReport,
    audio_path: Path,
    judge: LLMJudge,
    disagree_threshold: float | None = None,
    pad_seconds: float | None = None,
    work_dir: Path | None = None,
) -> ReviewReport:
    """Run the judge over flagged segments and attach verdicts to the report.

    Deterministic stages never change; only segments the disagreement detector
    flags are sent to the LLM. Verdicts are stored on each `AlignedSegment` as
    `llm_judgment`. Returns the same report object (mutated).
    """
    flagged = select_for_judgment(report, disagree_threshold)
    if not flagged:
        return report

    pad = config.JUDGE_PAD_SECONDS if pad_seconds is None else pad_seconds
    cleanup = None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="approach2_judge_"))
        cleanup = work_dir

    try:
        for segment in flagged:
            span_start = max(0.0, segment.start - pad)
            span_end = segment.end + pad
            clip = extract_span(audio_path, span_start, span_end, work_dir)
            verdict = judge.judge(
                JudgeRequest(segment, audio_bytes=clip.read_bytes(), mime_type="audio/wav")
            )
            if verdict is not None:
                segment.llm_judgment = verdict
    finally:
        if cleanup is not None:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)
    return report

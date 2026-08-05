"""Deterministic signals for gold vs candidate comparison.

These are cheap, reproducible metrics computed WITHOUT an LLM. They back the
overall score and help decide where the LLM classifier should look.

All lexical signals operate on normalized text (see normalize.py) so that
formatting noise ("$50" vs "fifty dollars") does not count as an error.
"""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from approach_1.src.normalize import normalize_number_words, normalize_text

# ---------------------------------------------------------------------------
# Lexical signals
# ---------------------------------------------------------------------------

def _norm_tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def word_error_rate(gold: str, candidate: str) -> float:
    """Levenshtein WER over normalized word tokens, in [0, 1]."""
    g, c = _norm_tokens(gold), _norm_tokens(candidate)
    return Levenshtein.distance(g, c) / max(1, len(g))


def char_error_rate(gold: str, candidate: str) -> float:
    """Levenshtein CER over normalized characters, in [0, 1]."""
    g, c = normalize_text(gold), normalize_text(candidate)
    return Levenshtein.distance(g, c) / max(1, len(g))


def token_coverage(gold: str, candidate: str) -> float:
    """Fraction of gold tokens present in the candidate (set-based recall)."""
    g = set(_norm_tokens(gold))
    c = set(_norm_tokens(candidate))
    if not g:
        return 1.0
    return len(g & c) / len(g)


def hallucination_ratio(gold: str, candidate: str) -> float:
    """Fraction of candidate tokens absent from gold (set-based precision gap)."""
    g = set(_norm_tokens(gold))
    c = set(_norm_tokens(candidate))
    if not c:
        return 0.0
    return len(c - g) / len(c)


# ---------------------------------------------------------------------------
# Entity extraction (numeric + calendar; pluggable for full NER later)
# ---------------------------------------------------------------------------

_TIME_PAT = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(?:am|pm)\b|\b(\d{1,2}):(\d{2})\b", re.IGNORECASE)
_NUM_UNIT_PAT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:milligrams?|grams?|kilograms?|mg|g|kg|ml|liters?|l|"
    r"meters?|centimeters?|millimeters?|m|cm|mm|dollars?|euros?|pounds?|percent)\b",
    re.IGNORECASE,
)
_CURRENCY_PAT = re.compile(r"[$€£¥]\s*\d+(?:\.\d+)?\b")
_DATE_PAT = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_PAT = re.compile(
    r"\b(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|"
    r"Sat(?:urday)?|Sun(?:day)?)\b"
)
_MONTH_PAT = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"
)
_YEAR_PAT = re.compile(r"\b\d{4}\b")
_NUM_PAT = re.compile(r"\b\d+\b")

_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


def _canon_time(hours: int, minutes: int, meridiem: str) -> str:
    h = hours
    if meridiem == "pm" and h < 12:
        h += 12
    elif meridiem == "am" and h == 12:
        h = 0
    return f"{h:02d}:{minutes:02d}"


def _canonical_entity(match: str) -> str:
    tm = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap]m)", match, re.IGNORECASE)
    if tm:
        return _canon_time(int(tm.group(1)), int(tm.group(2) or 0), tm.group(3).lower())
    tm2 = re.fullmatch(r"(\d{1,2}):(\d{2})", match)
    if tm2:
        return _canon_time(int(tm2.group(1)), int(tm2.group(2)), "")
    return normalize_text(_ORDINAL_RE.sub(r"\1", match))


def extract_entities(text: str) -> list[str]:
    """Return canonical numeric/calendar entities present in the raw text."""
    surface = normalize_number_words(text)
    found: list[str] = []
    for pattern in (_TIME_PAT, _NUM_UNIT_PAT, _CURRENCY_PAT, _DATE_PAT, _DAY_PAT, _MONTH_PAT, _YEAR_PAT, _NUM_PAT):
        found.extend(m.group(0) for m in pattern.finditer(surface))

    canon = sorted({_canonical_entity(m) for m in found}, key=len, reverse=True)
    kept: list[str] = []
    for c in canon:
        if any(c in k for k in kept):
            continue
        kept.append(c)
    return sorted(kept)


def entity_recall(gold: str, candidate: str) -> float:
    g = set(extract_entities(gold))
    c = set(extract_entities(candidate))
    if not g:
        return 1.0
    return len(g & c) / len(g)


def entity_precision(gold: str, candidate: str) -> float:
    g = set(extract_entities(gold))
    c = set(extract_entities(candidate))
    if not c:
        return 1.0
    return len(g & c) / len(c)


# ---------------------------------------------------------------------------
# Semantic signal (optional, lazy-loaded sentence-transformers)
# ---------------------------------------------------------------------------

class SentenceEmbedder:
    """Lazy singleton wrapper around a local sentence-transformers model."""

    _model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def _load(self) -> None:
        if SentenceEmbedder._model is None:
            from sentence_transformers import SentenceTransformer

            SentenceEmbedder._model = SentenceTransformer(self.model_name)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of normalized sentence embeddings, in [0, 1]."""
        import numpy as np

        self._load()
        emb = SentenceEmbedder._model.encode([normalize_text(a), normalize_text(b)])
        a_vec, b_vec = emb[0], emb[1]
        norm = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
        if norm == 0:
            return 0.0
        cos = float(np.dot(a_vec, b_vec) / norm)
        return max(0.0, min(1.0, cos))


def semantic_similarity(gold: str, candidate: str, embedder=None) -> float | None:
    """Return embedding cosine similarity, or None when no embedder is available."""
    if embedder is None:
        return None
    try:
        return embedder.similarity(gold, candidate)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_signals(gold: str, candidate: str, embedder=None) -> dict:
    """Compute the full signal set for the report."""
    signals = {
        "wer": round(word_error_rate(gold, candidate), 4),
        "cer": round(char_error_rate(gold, candidate), 4),
        "coverage": round(token_coverage(gold, candidate), 4),
        "hallucination_ratio": round(hallucination_ratio(gold, candidate), 4),
        "entity_recall": round(entity_recall(gold, candidate), 4),
        "entity_precision": round(entity_precision(gold, candidate), 4),
    }
    sem = semantic_similarity(gold, candidate, embedder)
    signals["semantic_similarity"] = round(sem, 4) if sem is not None else None
    return signals

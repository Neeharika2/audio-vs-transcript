"""Segmentation and alignment of gold vs candidate transcripts.

Segments both documents into comparable units (sentences, or fixed-width
windows when the STT output has no punctuation), then aligns them with
Needleman-Wunsch over a hybrid similarity score (fuzzy char match + optional
sentence embeddings). Output: aligned pairs plus unmatched gold/candidate
segments, which drive the missing / hallucinated categories downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - rapidfuzz is a declared dependency
    fuzz = None

from approach_1.src.normalize import normalize_text

MATCH_THRESHOLD = 0.60

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[“])")
_ABBREVIATION_RE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|vs|etc|approx|fig)\.", re.IGNORECASE)
_SENTINEL = "\x00"


@dataclass
class Segment:
    text: str
    norm: str = field(default="")


@dataclass
class AlignedPair:
    gold: Segment
    candidate: Segment
    similarity: float


@dataclass
class AlignmentResult:
    pairs: list[AlignedPair]
    unmatched_gold: list[Segment]
    unmatched_candidate: list[Segment]
    covered_gold: list[Segment] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        return {
            "gold_segments": len(self.pairs) + len(self.unmatched_gold) + len(self.covered_gold),
            "candidate_segments": len(self.pairs) + len(self.unmatched_candidate),
            "matched": len(self.pairs),
            "unmatched_gold": len(self.unmatched_gold),
            "unmatched_candidate": len(self.unmatched_candidate),
            "covered_gold": len(self.covered_gold),
        }

# Segmentation
def segment_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    protected = _ABBREVIATION_RE.sub(lambda m: m.group(0)[:-1] + _SENTINEL, text)
    parts = _SENTENCE_RE.split(protected.strip())
    return [p.replace(_SENTINEL, ".").strip() for p in parts if p.strip()]


def segment_windows(text: str, window: int = 40, overlap: int = 10) -> list[str]:
    """Fallback segmentation for unpunctuated text: sliding word windows."""
    words = text.strip().split()
    if not words:
        return []
    if len(words) <= window:
        return [" ".join(words)]
    step = max(1, window - overlap)
    return [
        " ".join(words[i : i + window])
        for i in range(0, max(1, len(words) - window + 1), step)
    ]


def segment(text: str, window: int = 40, overlap: int = 10) -> list[Segment]:
    """Segment text into comparable units with normalized forms."""
    sentences = segment_sentences(text)
    if len(sentences) > 1:
        chunks = sentences
    else:
        chunks = segment_windows(text, window=window, overlap=overlap)
    return [Segment(text=c, norm=normalize_text(c)) for c in chunks]


# Similarity
def _char_similarity(a: str, b: str) -> float:
    if fuzz is None:
        return 1.0 if a == b else 0.0
    if not a or not b:
        return 0.0
    return max(
        fuzz.ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ) / 100.0


def similarity(a: Segment, b: Segment, embedder=None) -> float:
    """Combined char + (optional) embedding similarity in [0, 1]."""
    char = _char_similarity(a.norm, b.norm)
    if embedder is None:
        return char
    try:
        emb = embedder.similarity(a.norm, b.norm)
    except Exception:
        return char
    return max(char, emb)


# Alignment

def _greedy_match(gold: list[Segment], cand: list[Segment], embedder=None):
    """Greedy bipartite matching. Handles reordering; pairs are 1:1."""
    n, m = len(gold), len(cand)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    sims = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            sims[i][j] = similarity(gold[i], cand[j], embedder)

    candidates: list[tuple[float, int, int]] = [
        (sims[i][j], i, j) for i in range(n) for j in range(m)
        if sims[i][j] >= MATCH_THRESHOLD
    ]
    candidates.sort(reverse=True)

    pairs: list[tuple[int, int, float]] = []
    used_gold: set[int] = set()
    used_cand: set[int] = set()
    for sim, i, j in candidates:
        if i in used_gold or j in used_cand:
            continue
        pairs.append((i, j, sim))
        used_gold.add(i)
        used_cand.add(j)

    unmatched_gold = [i for i in range(n) if i not in used_gold]
    unmatched_cand = [j for j in range(m) if j not in used_cand]
    return pairs, unmatched_gold, unmatched_cand


def align(
    gold_text: str,
    candidate_text: str,
    embedder=None,
) -> AlignmentResult:
    """Align gold and candidate transcripts into pairs + unmatched segments."""
    gold = segment(gold_text)
    cand = segment(candidate_text)
    pairs_idx, gold_unmatched_idx, cand_unmatched_idx = _greedy_match(gold, cand, embedder)

    pairs: list[AlignedPair] = [
        AlignedPair(gold=gold[i], candidate=cand[j], similarity=sim)
        for i, j, sim in pairs_idx
    ]

    # Post-pass: gold segments left unmatched may be contained inside a
    # matched candidate segment (merged by the STT). Downgrade those to
    # "covered" instead of "missing" so the classifier isn't misled.
    covered: list[Segment] = []
    truly_unmatched_gold: list[Segment] = []
    for i in sorted(set(gold_unmatched_idx)):
        g = gold[i]
        best = max((_char_similarity(g.norm, p.candidate.norm) for p in pairs), default=0.0)
        if best >= MATCH_THRESHOLD:
            covered.append(g)
        else:
            truly_unmatched_gold.append(g)

    unmatched_cand = [cand[j] for j in sorted(set(cand_unmatched_idx))]

    return AlignmentResult(
        pairs=pairs,
        unmatched_gold=truly_unmatched_gold,
        unmatched_candidate=unmatched_cand,
        covered_gold=covered,
    )

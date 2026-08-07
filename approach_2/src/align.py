"""Segment alignment: time-overlap + text similarity, Needleman-Wunsch, 1:N merge.

Two engines rarely produce the same segmentation: they split and merge sentences
differently, timestamps drift, and fillers get dropped. Aligning by timestamps
alone or by text alone fails, so we combine both into a score matrix and run
Needleman-Wunsch over the two segment sequences. A post-pass folds unmatched
adjacent segments into matched pairs to absorb 1:N splits/merges.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from approach_2.src.normalize import normalize_text
from approach_2 import config
from approach_2.src.models import AlignedSegment, EngineSegment

# Filler words are stripped only for matching/agreement; stored text is never
# mutated. Reused by compare.py.
_FILLERS = {"uh", "um", "uhh", "umm", "hmm", "mmm", "hm", "mm", "er", "ah", "oh"}


def norm_words(seg: EngineSegment) -> list[str]:
    """Normalized, filler-stripped word tokens used for matching and agreement."""
    return [w for w in normalize_text(seg.text).split() if w not in _FILLERS]


def norm_text(seg: EngineSegment) -> str:
    """Normalized, filler-stripped text used for matching and agreement."""
    return " ".join(norm_words(seg))


def _overlap(a: EngineSegment, b: EngineSegment) -> float:
    """Shared time / min duration, in [0, 1]; 0 when disjoint."""
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    overlap = max(0.0, end - start)
    dur = min(a.end - a.start, b.end - b.start)
    if dur <= 0:
        return 1.0 if overlap > 0 else 0.0
    return overlap / dur


def _text_sim(a: str, b: str) -> float:
    """Positional similarity with a length penalty, in [0, 1].

    Used both for the match score and merge acceptance. The length penalty is
    what lets text discriminate a 1:N split: every constituent of a merged
    sentence passes token_set_ratio at ~1.0 (it is a subset), so a positional
    ratio scaled by min/max length is needed to prefer the longest, best-fit
    constituent and to reject absorbing a dropped segment.
    """
    if not a or not b:
        return 0.0
    sim = fuzz.ratio(a, b) / 100.0
    len_a, len_b = len(a.split()), len(b.split())
    length_factor = min(len_a, len_b) / max(len_a, len_b)
    return sim * length_factor


def score(a: EngineSegment, b: EngineSegment) -> float:
    """Combined match score in [0, 1]: half time overlap, half text similarity."""
    return 0.5 * _overlap(a, b) + 0.5 * _text_sim(norm_text(a), norm_text(b))


def _nw_pairs(segs_a: list[EngineSegment], segs_b: list[EngineSegment]) -> list[tuple[int, int]]:
    """Needleman-Wunsch over the score matrix -> optimal 1:1 index pairs.

    Each cell contributes (score - MATCH_THRESHOLD) when paired, so only
    genuinely similar segments beat taking a gap.
    """
    n, m = len(segs_a), len(segs_b)
    if n == 0 or m == 0:
        return []
    scores = [[score(segs_a[i], segs_b[j]) for j in range(m)] for i in range(n)]
    threshold = config.MATCH_THRESHOLD

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            diag = dp[i][j] + scores[i][j] - threshold
            up = dp[i][j + 1]
            left = dp[i + 1][j]
            dp[i + 1][j + 1] = max(diag, up, left)

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + scores[i - 1][j - 1] - threshold
        up = dp[i - 1][j]
        left = dp[i][j - 1]
        if dp[i][j] == diag and diag >= up and diag >= left:
            if scores[i - 1][j - 1] >= threshold:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == up and up >= left:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _merge_pass(
    pairs: list[tuple[int, int]],
    segs_a: list[EngineSegment],
    segs_b: list[EngineSegment],
) -> tuple[list[tuple[list[int], list[int]]], list[int], list[int]]:
    """Fold unmatched adjacent segments into matched pairs (1:N split/merge).

    Returns (merged_pairs, unmatched_a, unmatched_b) where each merged pair is
    (a_indices, b_indices). A segment is only absorbed when the combined
    normalized text still matches the counterpart above MATCH_THRESHOLD.
    """
    n, m = len(segs_a), len(segs_b)
    threshold = config.MATCH_THRESHOLD
    used_a = {i for i, _ in pairs}
    used_b = {j for _, j in pairs}
    groups: list[tuple[set[int], set[int]]] = [({i}, {j}) for i, j in pairs]

    def try_add(a_idx: set[int], b_idx: set[int], cand_a: int | None, cand_b: int | None) -> bool:
        new_a = a_idx | ({cand_a} if cand_a is not None else set())
        new_b = b_idx | ({cand_b} if cand_b is not None else set())
        ta = " ".join(norm_text(segs_a[k]) for k in sorted(new_a))
        tb = " ".join(norm_text(segs_b[k]) for k in sorted(new_b))
        return _text_sim(ta, tb) >= threshold

    for a_idx, b_idx in groups:
        changed = True
        while changed:
            changed = False
            for cand in (min(a_idx) - 1, max(a_idx) + 1):
                if len(a_idx) >= 3 or cand < 0 or cand >= n or cand in used_a:
                    continue
                if try_add(a_idx, b_idx, cand, None):
                    a_idx.add(cand)
                    used_a.add(cand)
                    changed = True
                    break
            if changed:
                continue
            for cand in (min(b_idx) - 1, max(b_idx) + 1):
                if len(b_idx) >= 3 or cand < 0 or cand >= m or cand in used_b:
                    continue
                if try_add(a_idx, b_idx, None, cand):
                    b_idx.add(cand)
                    used_b.add(cand)
                    changed = True
                    break

    merged = [(sorted(a_idx), sorted(b_idx)) for a_idx, b_idx in groups]
    unmatched_a = sorted(set(range(n)) - used_a)
    unmatched_b = sorted(set(range(m)) - used_b)
    return merged, unmatched_a, unmatched_b


def _merge_engine_segments(segs: list[EngineSegment]) -> EngineSegment | None:
    """Combine constituent segments of one engine into a single EngineSegment."""
    if not segs:
        return None
    words = [w for s in segs for w in s.words]
    confs = [w.confidence for w in words if w.confidence is not None]
    confidence = round(sum(confs) / len(confs), 4) if confs else None
    return EngineSegment(
        engine=segs[0].engine,
        start=min(s.start for s in segs),
        end=max(s.end for s in segs),
        text=" ".join(s.text for s in segs),
        confidence=confidence,
        words=words,
    )


def align(segs_a: list[EngineSegment], segs_b: list[EngineSegment]) -> list[AlignedSegment]:
    """Align two engine segment streams into time-ordered AlignedSegments."""
    pairs = _nw_pairs(segs_a, segs_b)
    merged, unmatched_a, unmatched_b = _merge_pass(pairs, segs_a, segs_b)

    aligned: list[AlignedSegment] = []
    for a_idx, b_idx in merged:
        a_seg = _merge_engine_segments([segs_a[i] for i in a_idx])
        b_seg = _merge_engine_segments([segs_b[j] for j in b_idx])
        aligned.append(
            AlignedSegment(
                idx=len(aligned),
                start=min(s.start for s in (a_seg, b_seg) if s is not None),
                end=max(s.end for s in (a_seg, b_seg) if s is not None),
                engine_a=a_seg,
                engine_b=b_seg,
                agreement=0.0,
            )
        )
    for i in unmatched_a:
        s = segs_a[i]
        aligned.append(
            AlignedSegment(idx=len(aligned), start=s.start, end=s.end, engine_a=s, engine_b=None, agreement=0.0)
        )
    for j in unmatched_b:
        s = segs_b[j]
        aligned.append(
            AlignedSegment(idx=len(aligned), start=s.start, end=s.end, engine_a=None, engine_b=s, agreement=0.0)
        )

    aligned.sort(key=lambda s: s.start)
    for k, s in enumerate(aligned):
        s.idx = k
    return aligned

"""Segment alignment: global word-level alignment reassembled into segments.

Two engines rarely produce the same segmentation: they split and merge
sentences differently, timestamps drift, and fillers get dropped. Comparing
raw segment texts is wrong because the boundaries themselves disagree — a
segment that engine A produced in two parts may be one engine B segment, so
text from a neighbouring spoken span leaks into the comparison. Instead of
pairing raw segments, we flatten both engines to normalized word streams,
align them globally with Needleman-Wunsch (word order is identical even when
segment boundaries are not; matches far apart in time are penalized so a
repeated phrase cannot be pulled across a dropped span), and group the aligned
words back into AlignedSegments. Each engine's segment is assigned whole to
the window that holds its matched words, so the two sides of every comparison
cover the same spoken content regardless of how each engine happened to cut it.
"""

from __future__ import annotations

from approach_2.src.normalize import normalize_text
from approach_2.src.models import AlignedSegment, EngineSegment

# Filler words are stripped only for matching/agreement; stored text is never
# mutated. Reused by compare.py.
_FILLERS = {"uh", "um", "uhh", "umm", "hmm", "mmm", "hm", "mm", "er", "ah", "oh"}

# Word-level matches whose (interpolated) timestamps differ by more than
# _TIME_TOLERANCE seconds are penalized _TIME_PENALTY per extra second. STT
# boundaries drift by well under a second for identical content, so this keeps
# the alignment from smearing a repeated phrase (e.g. "part of the story") to a
# different time region when one engine drops a segment.
_TIME_TOLERANCE = 1.5
_TIME_PENALTY = 2.0


def norm_words(seg: EngineSegment) -> list[str]:
    """Normalized, filler-stripped word tokens used for matching and agreement."""
    return [w for w in normalize_text(seg.text).split() if w not in _FILLERS]


def norm_text(seg: EngineSegment) -> str:
    """Normalized, filler-stripped text used for matching and agreement."""
    return " ".join(norm_words(seg))


def _word_align(
    tokens_a: list[str],
    times_a: list[float],
    tokens_b: list[str],
    times_b: list[float],
) -> list[tuple[int, int]]:
    """Global Needleman-Wunsch over word tokens -> optimal (i, j) correspondences.

    Both engines transcribe the same audio, so word order is identical and the
    global optimum is the correct content mapping. Equal words score +1,
    substitutions and gaps score -1; a diagonal move is additionally penalized
    when its interpolated timestamps are far apart, so a repeated phrase cannot
    be dragged to a different spoken span just because the text matches. Every
    non-gap cell (match or substitution) is kept as a correspondence, anchoring
    where each spoken word lands on the other side.
    """
    n, m = len(tokens_a), len(tokens_b)
    if n == 0 or m == 0:
        return []

    def _diag(i: int, j: int) -> float:
        dt = abs(times_a[i - 1] - times_b[j - 1])
        time_cost = _TIME_PENALTY * max(0.0, dt - _TIME_TOLERANCE)
        return dp[i - 1][j - 1] + (1 if tokens_a[i - 1] == tokens_b[j - 1] else -1) - time_cost

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = -i
    for j in range(m + 1):
        dp[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(_diag(i, j), dp[i - 1][j] - 1, dp[i][j - 1] - 1)

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        diag = _diag(i, j) if i > 0 and j > 0 else float("-inf")
        up = dp[i - 1][j] - 1 if i > 0 else float("-inf")
        left = dp[i][j - 1] - 1 if j > 0 else float("-inf")
        if i > 0 and j > 0 and diag >= up and diag >= left:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and up >= left:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


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
    """Align two engine streams into time-ordered AlignedSegments that each
    compare the same spoken content.

    1. Flatten both engines to normalized word streams and align them globally
       at word level (order is identical; segment boundaries are not).
    2. Open one window per engine-A segment, then merge adjacent windows when a
       single engine-B segment contributes words to both — engine B spans the
       boundary, so the split is only engine A's. This absorbs 1:N, N:1, and
       crossing (2x2) splits without choosing either engine as a reference.
    3. Assign every engine-B segment whole to the window holding its matched
       words. Segments with no matched words go to the best time-overlapping
       window, or become engine-B-only segments when they cover audio engine A
       never transcribed (genuine unmatched content).
    4. Word-level comparison happens later in compare(), per window, on these
       already-aligned spoken spans.
    """
    tokens_a: list[str] = []
    seg_of_a: list[int] = []
    times_a: list[float] = []
    for si, seg in enumerate(segs_a):
        words = norm_words(seg)
        span = seg.end - seg.start
        for k, word in enumerate(words):
            tokens_a.append(word)
            seg_of_a.append(si)
            times_a.append(seg.start + (k + 0.5) / max(1, len(words)) * span)

    tokens_b: list[str] = []
    seg_of_b: list[int] = []
    times_b: list[float] = []
    for si, seg in enumerate(segs_b):
        words = norm_words(seg)
        span = seg.end - seg.start
        for k, word in enumerate(words):
            tokens_b.append(word)
            seg_of_b.append(si)
            times_b.append(seg.start + (k + 0.5) / max(1, len(words)) * span)

    pairs = _word_align(tokens_a, times_a, tokens_b, times_b)

    matched_b: list[set[int]] = [set() for _ in segs_a]
    for ia, ib in pairs:
        matched_b[seg_of_a[ia]].add(seg_of_b[ib])

    windows: list[tuple[list[int], set[int]]] = [([i], matched_b[i]) for i in range(len(segs_a))]

    # Merge adjacent windows sharing an engine-B segment until no boundary is
    # crossed by a single engine-B segment anymore.
    while True:
        merged: list[tuple[list[int], set[int]]] = []
        changed = False
        k = 0
        while k < len(windows):
            a_idx, b_set = windows[k]
            nxt = k + 1
            while nxt < len(windows) and b_set & windows[nxt][1]:
                a_idx = a_idx + windows[nxt][0]
                b_set = b_set | windows[nxt][1]
                nxt += 1
            if nxt > k + 1:
                changed = True
            merged.append((a_idx, b_set))
            k = nxt
        windows = merged
        if not changed:
            break

    b_to_window: dict[int, int] = {}
    for wi, (_, b_set) in enumerate(windows):
        for bi in b_set:
            b_to_window.setdefault(bi, wi)

    for bi, seg in enumerate(segs_b):
        if bi in b_to_window:
            continue
        best, best_overlap = None, 0.0
        for wi, (a_idx, _) in enumerate(windows):
            start = min(segs_a[i].start for i in a_idx)
            end = max(segs_a[i].end for i in a_idx)
            overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
            if overlap > best_overlap:
                best, best_overlap = wi, overlap
        if best is not None and best_overlap > 0:
            b_to_window[bi] = best

    aligned: list[AlignedSegment] = []
    for wi, (a_idx, _) in enumerate(windows):
        a_seg = _merge_engine_segments([segs_a[i] for i in a_idx])
        b_idx = sorted(bi for bi, w in b_to_window.items() if w == wi)
        b_seg = _merge_engine_segments([segs_b[bi] for bi in b_idx]) if b_idx else None
        aligned.append(
            AlignedSegment(idx=len(aligned), start=a_seg.start, end=a_seg.end, engine_a=a_seg, engine_b=b_seg, agreement=0.0)
        )
    for bi, seg in enumerate(segs_b):
        if bi in b_to_window:
            continue
        aligned.append(
            AlignedSegment(idx=len(aligned), start=seg.start, end=seg.end, engine_a=None, engine_b=seg, agreement=0.0)
        )

    aligned.sort(key=lambda s: s.start)
    for k, seg in enumerate(aligned):
        seg.idx = k
    return aligned

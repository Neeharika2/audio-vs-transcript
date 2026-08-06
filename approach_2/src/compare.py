"""Word-level diff and agreement for aligned segments."""

from __future__ import annotations

from approach_2.src.align import norm_words
from approach_2.src.models import AlignedSegment, WordOp


def word_diff(a_words: list[str], b_words: list[str]) -> list[WordOp]:
    """Levenshtein alignment over word tokens -> match/substitute/insert/delete ops.

    `a_words` is engine A's side, `b_words` engine B's side. A delete is a word
    only engine A heard; an insert only engine B heard; a substitute is a word
    the two engines heard differently.
    """
    n, m = len(a_words), len(b_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a_words[i - 1] == b_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: list[WordOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_words[i - 1] == b_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(WordOp(text=a_words[i - 1], op="match"))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(WordOp(text=b_words[j - 1], op="substitute"))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(WordOp(text=a_words[i - 1], op="delete"))
            i -= 1
        else:
            ops.append(WordOp(text=b_words[j - 1], op="insert"))
            j -= 1
    ops.reverse()
    return ops


def agreement(diff: list[WordOp], len_a: int, len_b: int) -> float:
    """1 - WER over normalized words; 1.0 when both sides are empty."""
    if max(len_a, len_b) == 0:
        return 1.0
    errors = sum(1 for op in diff if op.op != "match")
    return round(1.0 - errors / max(len_a, len_b), 4)


def compare(seg: AlignedSegment) -> AlignedSegment:
    """Fill agreement + diff for a segment; missing side => agreement 0.0."""
    if seg.engine_a is None or seg.engine_b is None:
        seg.agreement = 0.0
        seg.diff = []
        return seg
    a_words = norm_words(seg.engine_a)
    b_words = norm_words(seg.engine_b)
    seg.diff = word_diff(a_words, b_words)
    seg.agreement = agreement(seg.diff, len(a_words), len(b_words))
    return seg

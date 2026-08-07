"""Word-level diff and agreement for aligned segments."""

from __future__ import annotations

from approach_2.src.align import norm_words
from approach_2.src.models import AlignedSegment, TokenOp, WordOp


def token_align(a_words: list[str], b_words: list[str]) -> list[TokenOp]:
    """Levenshtein alignment over word tokens -> per-token (a, b) ops.

    `a_words` is engine A's side, `b_words` engine B's side. This is the single
    shared backtrack for every stage that needs to explain *why* two sides
    differ (diff + judge). A delete is a word only engine A heard (`b=None`),
    an insert only engine B heard (`a=None`), a substitute a word the two
    engines heard differently. Consumers derive their own view from these ops.
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

    ops: list[TokenOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a_words[i - 1] == b_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(TokenOp(op="match", a=a_words[i - 1], b=b_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(TokenOp(op="substitute", a=a_words[i - 1], b=b_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(TokenOp(op="delete", a=a_words[i - 1]))
            i -= 1
        else:
            ops.append(TokenOp(op="insert", b=b_words[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def word_diff(a_words: list[str], b_words: list[str]) -> list[WordOp]:
    """Levenshtein alignment -> match/substitute/insert/delete ops for display."""
    return [
        WordOp(text=op.a if op.op == "delete" else op.b, op=op.op)
        for op in token_align(a_words, b_words)
    ]


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

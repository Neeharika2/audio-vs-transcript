"""Spot-check sampling and acceptance.

The reviewer verifies the sampled segments against the audio; the sample is
everything already flagged (mandatory + disagreements) plus a seeded random
fraction of the rest, so it is both risk-concentrated and reproducible.
"""

from __future__ import annotations

import random

from approach_2 import config
from approach_2.src.models import AlignedSegment

EXPAND_THRESHOLD = 0.95


def sample_review_set(
    segments: list[AlignedSegment],
    seed: int | None = None,
    fraction: float | None = None,
) -> list[int]:
    """Segment indices to review, highest risk first."""
    seed = config.SPOT_CHECK_SEED if seed is None else seed
    fraction = config.SPOT_CHECK_FRACTION if fraction is None else fraction
    disagree_threshold = config.DISAGREE_THRESHOLD

    mandatory = [s.idx for s in segments if s.tier == "mandatory"]
    disagreements = [
        s.idx for s in segments if s.agreement is None or s.agreement < disagree_threshold
    ]
    flagged = mandatory + [i for i in disagreements if i not in mandatory]
    rest = [s.idx for s in segments if s.idx not in flagged]

    rng = random.Random(seed)
    if rest:
        k = min(max(1, round(len(rest) * fraction)), len(rest))
        random_sample = rng.sample(rest, k)
    else:
        random_sample = []
    return flagged + random_sample


def acceptance_check(accuracy: float | None) -> tuple[bool, bool]:
    """Returns (accepted, expanded). Accuracy None means not yet reviewed."""
    if accuracy is None:
        return False, False
    if accuracy >= config.SPOT_CHECK_ACCEPT:
        return True, False
    if accuracy >= EXPAND_THRESHOLD:
        return False, True
    return False, False

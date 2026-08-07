"""Spot-check sampling, acceptance, and reviewer-verdict merging.

The reviewer verifies segments against the audio. The sample is everything
already flagged (mandatory + disagreements) plus a seeded random fraction of
the rest, so it is both risk-concentrated and reproducible. Reviewer verdicts
are merged back into the report, which computes the sample accuracy and turns
it into an accept / expand / full-review decision.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from approach_2 import config
from approach_2.src.align import norm_words
from approach_2.src.models import AlignedSegment, ReviewReport

EXPAND_THRESHOLD = 0.95


def load_glossary(path: str = "") -> set[str]:
    """Load the domain-term wordlist as lowercase tokens; empty when disabled."""
    path = path or config.GLOSSARY_PATH
    if not path:
        return set()
    glossary_path = Path(path)
    if not glossary_path.is_file():
        return set()
    terms: set[str] = set()
    for line in glossary_path.read_text(encoding="utf-8").splitlines():
        terms.update(w for w in re.findall(r"[a-z0-9']+", line.lower()) if w)
    return terms


def _contains_glossary(seg: AlignedSegment, glossary: set[str]) -> bool:
    for side in (seg.engine_a, seg.engine_b):
        if side is not None and glossary.intersection(norm_words(side)):
            return True
    return False


def sample_review_set(
    segments: list[AlignedSegment],
    seed: int | None = None,
    fraction: float | None = None,
) -> list[int]:
    """Segment indices to review, highest risk first.

    Mandatory and disagreement segments always come back. A review_technical
    segment is also escalated when it contains a glossary term (the tier rule);
    otherwise it is only eligible for the random slice.
    """
    seed = config.SPOT_CHECK_SEED if seed is None else seed
    fraction = config.SPOT_CHECK_FRACTION if fraction is None else fraction
    disagree_threshold = config.REVIEW_DISAGREE_THRESHOLD

    mandatory = [s.idx for s in segments if s.tier == "mandatory"]
    glossary = load_glossary()
    if glossary:
        mandatory += [
            s.idx
            for s in segments
            if s.tier == "review_technical" and _contains_glossary(s, glossary)
        ]
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


def apply_review(report: ReviewReport, verdicts: dict[int, dict]) -> ReviewReport:
    """Merge reviewer verdicts/corrections into a report and recompute acceptance.

    `verdicts` maps segment index to {"verdict": "correct"|"incorrect", ...}.
    Accuracy is correct / reviewed across every marked segment; the acceptance
    thresholds from `acceptance_check` decide the outcome.
    """
    correct = reviewed = 0
    for seg in report.segments:
        entry = verdicts.get(seg.idx)
        if not entry:
            continue
        seg.correction = entry.get("correction")
        verdict = entry.get("verdict")
        if verdict not in ("correct", "incorrect"):
            continue
        seg.verdict = verdict
        reviewed += 1
        correct += verdict == "correct"

    accuracy = round(correct / reviewed, 4) if reviewed else None
    accepted, expanded = acceptance_check(accuracy)
    report.spot_check.accuracy = accuracy
    report.spot_check.accepted = accepted if reviewed else None
    report.spot_check.expanded = expanded
    report.spot_check.full_review = bool(reviewed) and not accepted and not expanded
    return report

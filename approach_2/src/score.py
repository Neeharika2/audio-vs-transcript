"""Per-segment confidence and review tier."""

from __future__ import annotations

from approach_2 import config
from approach_2.src.models import AlignedSegment

# Fixed weights from the locked formula (docs/plan.md §5.5).
WEIGHT_ENGINE_CONF = 0.40
WEIGHT_AGREEMENT = 0.45
WEIGHT_LOW_CONF = 0.15


def _confidence_words(seg: AlignedSegment) -> list[float]:
    words = []
    for side in (seg.engine_a, seg.engine_b):
        if side is not None:
            words += [w.confidence for w in side.words if w.confidence is not None]
    return words


def engine_conf(seg: AlignedSegment) -> float:
    """Mean word confidence across both engines; 0.0 if none available."""
    confs = _confidence_words(seg)
    return sum(confs) / len(confs) if confs else 0.0


def low_conf_ratio(seg: AlignedSegment) -> float:
    """Fraction of confident words below LOW_CONF_THRESHOLD; 1.0 if none."""
    confs = _confidence_words(seg)
    if not confs:
        return 1.0
    return sum(1 for c in confs if c < config.LOW_CONF_THRESHOLD) / len(confs)


def segment_confidence(seg: AlignedSegment) -> float:
    """The single locked formula, scaled to [0, 100] and rounded."""
    agreement = seg.agreement if seg.agreement is not None else 0.0
    confidence = 100.0 * (
        WEIGHT_ENGINE_CONF * engine_conf(seg)
        + WEIGHT_AGREEMENT * agreement
        + WEIGHT_LOW_CONF * (1.0 - low_conf_ratio(seg))
    )
    return round(max(0.0, min(100.0, confidence)))


def assign_tier(confidence: float) -> str:
    """Exactly the user's rule: >=98 auto-accept, 90-97 review-if-technical, <90 mandatory."""
    if confidence >= config.TIER_AUTO_ACCEPT:
        return "auto_accept"
    if confidence >= config.TIER_REVIEW:
        return "review_technical"
    return "mandatory"


def score(seg: AlignedSegment) -> AlignedSegment:
    seg.confidence = segment_confidence(seg)
    seg.tier = assign_tier(seg.confidence)
    return seg

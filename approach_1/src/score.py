"""Scoring and aggregation of findings + signals into an overall report score."""

from __future__ import annotations

from approach_1.src.models import ErrorItem, ScoreBreakdown

SEVERITY_WEIGHT = {"low": 2.0, "medium": 5.0, "high": 10.0}

DEFAULT_THRESHOLD = 90.0


def error_penalty(findings: list[ErrorItem]) -> float:
    """Sum of severity-weighted penalties across all findings."""
    return sum(SEVERITY_WEIGHT.get(f.severity, SEVERITY_WEIGHT["medium"]) for f in findings)


def base_signal_score(signals: dict) -> float:
    """Combine deterministic signals into a 0-100 fidelity base score."""
    sem = signals.get("semantic_similarity")
    entity = signals.get("entity_recall", 0.0)
    lexical = 1.0 - signals.get("wer", 1.0)

    if sem is not None:
        return 100.0 * (0.4 * sem + 0.3 * entity + 0.3 * lexical)
    return 100.0 * (0.6 * entity + 0.4 * lexical)


def score_report(
    findings: list[ErrorItem],
    signals: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[ScoreBreakdown, int, str]:
    """Return (breakdown, overall_score, status)."""
    semantic = signals.get("semantic_similarity")
    entity = signals.get("entity_recall", 0.0)
    lexical = 1.0 - signals.get("wer", 1.0)

    base = base_signal_score(signals)
    penalty = error_penalty(findings)
    overall = max(0, round(base - penalty))

    has_high = any(f.severity == "high" for f in findings)
    status = "Mismatch" if (has_high or overall < threshold) else "Match"

    breakdown = ScoreBreakdown(
        semantic=round(semantic * 100, 1) if semantic is not None else None,
        entity=round(entity * 100, 1),
        lexical=round(lexical * 100, 1),
        error_penalty=round(penalty, 1),
    )
    return breakdown, overall, status

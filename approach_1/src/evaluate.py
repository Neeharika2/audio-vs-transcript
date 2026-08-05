"""End-to-end evaluation pipeline orchestrator (normalize -> align -> signals
-> classify -> score -> report)."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from approach_1.src.align import align
from approach_1.src.classify import classify
from approach_1.src.models import (
    AlignmentStats,
    ErrorItem,
    EvaluationInputs,
    EvaluationReportV2,
    Meta,
)
from approach_1.src.score import score_report
from approach_1.src.signals import compute_signals

Embedder = Callable[[str, str], float]


def _group_findings(findings: list[ErrorItem]) -> dict[str, list[ErrorItem]]:
    return {
        "missing_information": [f for f in findings if f.category == "missing_information"],
        "incorrect_information": [f for f in findings if f.category == "incorrect_information"],
        "conflicting_information": [f for f in findings if f.category == "conflicting_information"],
        "hallucinated_information": [f for f in findings if f.category == "hallucinated_information"],
    }


def evaluate(
    gold_transcript: str,
    candidate_transcript: str,
    judge=None,
    embedder=None,
    inputs: EvaluationInputs | None = None,
    threshold: float = 90.0,
) -> EvaluationReportV2:
    """Run the full pipeline and return a structured V2 evaluation report."""
    t0 = time.perf_counter()

    alignment = align(gold_transcript, candidate_transcript, embedder)
    signals = compute_signals(gold_transcript, candidate_transcript, embedder)
    findings, llm_calls = classify(alignment, judge, gold_transcript, candidate_transcript)
    breakdown, overall, status = score_report(findings, signals, threshold=threshold)

    grouped = _group_findings(findings)
    report = EvaluationReportV2(
        id=f"evl_{uuid.uuid4().hex[:8]}",
        inputs=inputs or EvaluationInputs(),
        alignment=AlignmentStats(**alignment.stats),
        signals=signals,
        score_breakdown=breakdown,
        overall_score=overall,
        status=status,
        meta=Meta(
            llm_calls=llm_calls,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
        **grouped,
    )
    return report

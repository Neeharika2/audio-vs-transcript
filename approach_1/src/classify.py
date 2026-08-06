"""LLM classification of alignment results into the four discrepancy categories.

Each aligned pair is judged by a small, targeted prompt. Unmatched segments
map deterministically to missing/hallucinated. A final global review pass
prunes false positives and catches cross-segment contradictions.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar, cast

from pydantic import BaseModel

from approach_1.src.align import AlignmentResult, AlignedPair
from approach_1.src.models import ErrorItem, FindingList, SegmentJudgement
from approach_1.src.normalize import normalize_text
from approach_1.src.signals import extract_entities

CATEGORY_MAP = {
    "missing": "missing_information",
    "incorrect": "incorrect_information",
    "conflict": "conflicting_information",
    "hallucination": "hallucinated_information",
}


JudgementT = TypeVar("JudgementT", bound=BaseModel)


class Judge(Protocol):
    def judge(self, prompt: str, schema: type[JudgementT]) -> JudgementT:
        """Send a classification prompt and return a validated structured output."""
        ...


class MockJudge:
    """Deterministic judge for offline tests.

    Returns `relationship`/`severity` for segment judgements. For the global
    review (FindingList) it echoes the preliminary findings back unchanged so
    pair-level findings survive the review pass.
    """

    def __init__(self, relationship: str = "match", severity: str = "low"):
        self.relationship = relationship
        self.severity = severity

    def judge(self, prompt: str, schema: type[JudgementT]) -> JudgementT:
        if schema is FindingList:
            match = re.search(r"\[.*\]", prompt, flags=re.S)
            if match:
                return cast(JudgementT, FindingList(findings=[ErrorItem(**item) for item in json.loads(match.group(0))]))
            return cast(JudgementT, FindingList(findings=[]))
        return cast(JudgementT, schema(relationship=self.relationship, explanation="mock judgement", severity=self.severity))


_PAIR_PROMPT = """You are auditing a speech-to-text transcript against a reference (gold) transcript.

GOLD (reference) segment:
{gold}

CANDIDATE segment:
{candidate}

CONTEXT around the gold segment:
{gold_context}

CONTEXT around the candidate segment:
{candidate_context}

Classify how the CANDIDATE segment relates to the GOLD segment. Choose exactly one relationship:

- match: the candidate faithfully conveys the gold meaning (paraphrase, different wording, or reordering is OK).
- incorrect: both cover the same fact but the candidate changes its meaning (wrong value, name, number, or unit).
- conflict: the candidate asserts the opposite of the gold.
- missing: the gold content is absent from the candidate segment.
- hallucination: the candidate content has no basis in the gold segment.

Rules:
- Rephrasing the same fact is "match", never "incorrect".
- severity: low = minor wording; medium = changed detail; high = changed number/name/fact or direct contradiction.
- explanation: one short sentence justifying your choice.
"""

_GLOBAL_REVIEW_PROMPT = """You are the final reviewer of an STT transcript audit.

GOLD transcript:
{gold}

CANDIDATE transcript:
{candidate}

PRELIMINARY FINDINGS (JSON list):
{findings_json}

Review the findings against the two transcripts, then:
1. Remove findings that are false positives, e.g. a "missing" item that is actually present elsewhere in the candidate, or a "hallucinated" item that is actually a paraphrase of gold.
2. Correct the category of any finding you disagree with. Categories are: missing_information, incorrect_information, conflicting_information, hallucinated_information.
3. Add any discrepancy the earlier pass missed (for example a contradiction between different parts of the candidate and the gold).

Return the final list of findings. Keep findings concise and evidence-backed.
"""


def _neighbor_context(text: str, needle: str, radius: int = 150) -> str:
    idx = text.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    return text[start:end].strip()


def heuristic_judge(pair: AlignedPair) -> SegmentJudgement:
    """Deterministic, LLM-free segment classification.

    Used as a fallback when no LLM judge is configured, so the pipeline still
    detects value changes (incorrect) and negations (conflict) offline.
    """
    gold_norm = set(normalize_text(pair.gold.text).split())
    cand_norm = set(normalize_text(pair.candidate.text).split())
    if "not" in cand_norm and "not" not in gold_norm:
        return SegmentJudgement(
            relationship="conflict",
            explanation="Candidate negates a fact stated in the gold transcript.",
            severity="high",
        )
    if set(extract_entities(pair.gold.text)) != set(extract_entities(pair.candidate.text)):
        return SegmentJudgement(
            relationship="incorrect",
            explanation="Entity values (numbers/dates) differ between gold and candidate.",
            severity="medium",
        )
    return SegmentJudgement(
        relationship="match",
        explanation="Candidate faithfully conveys the gold meaning.",
        severity="low",
    )


def _auto_missing(seg) -> ErrorItem:
    return ErrorItem(
        category="missing_information",
        reference_text=seg.text,
        generated_text=None,
        context=None,
        explanation="Gold segment has no counterpart in the candidate transcript.",
        severity="medium",
    )


def _auto_hallucination(seg) -> ErrorItem:
    return ErrorItem(
        category="hallucinated_information",
        reference_text=None,
        generated_text=seg.text,
        context=None,
        explanation="Candidate segment has no basis in the gold transcript.",
        severity="medium",
    )


def classify(
    alignment: AlignmentResult,
    judge: Judge | None,
    gold_full: str,
    candidate_full: str,
) -> tuple[list[ErrorItem], int]:
    """Classify alignment results into findings. Returns (findings, llm_calls).

    When `judge` is None, pairs are classified with deterministic heuristics
    (so the pipeline is functional offline); llm_calls stays 0.
    """
    findings: list[ErrorItem] = []
    llm_calls = 0

    for pair in alignment.pairs:
        if judge is not None:
            prompt = _PAIR_PROMPT.format(
                gold=pair.gold.text,
                candidate=pair.candidate.text,
                gold_context=_neighbor_context(gold_full, pair.gold.text),
                candidate_context=_neighbor_context(candidate_full, pair.candidate.text),
            )
            judgement = judge.judge(prompt, SegmentJudgement)
            llm_calls += 1
        else:
            judgement = heuristic_judge(pair)
        if judgement.relationship == "match":
            continue
        category = CATEGORY_MAP.get(judgement.relationship)
        if not category:
            continue
        findings.append(
            ErrorItem(
                category=category,
                reference_text=pair.gold.text,
                generated_text=pair.candidate.text,
                context=None,
                explanation=judgement.explanation,
                severity=judgement.severity,
                signal_evidence={"segment_similarity": round(pair.similarity, 3)},
            )
        )

    findings.extend(_auto_missing(seg) for seg in alignment.unmatched_gold)
    findings.extend(_auto_hallucination(seg) for seg in alignment.unmatched_candidate)

    if judge is not None and findings:
        llm_calls += 1
        review = judge.judge(
            _GLOBAL_REVIEW_PROMPT.format(
                gold=gold_full,
                candidate=candidate_full,
                findings_json=json.dumps([f.model_dump() for f in findings], indent=2),
            ),
            FindingList,
        )
        findings = list(review.findings)

    return findings, llm_calls

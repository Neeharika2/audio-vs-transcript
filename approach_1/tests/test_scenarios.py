"""Scenario-driven tests for Approach 1 (gold vs candidate STT).

Layer 1 (deterministic, no API): the offline heuristic evaluator must emit
findings exactly for the cases a heuristic can see (missing / negation /
number change). Scenarios a heuristic cannot see (garbled technical word) must
produce NO finding -- that is exactly why an LLM judge exists; it is not a bug.
Layer 2 (fake judge, no API): the pipeline must correctly consume an LLM
verdict and map it into the right finding bucket.

Layer 3 (real DeepSeek/Gemini) is NOT implemented here; it is optional,
API-key gated, and measures aggregate accuracy rather than exact equality.
"""

import pytest

from approach_1.src.align import align
from approach_1.src.classify import classify
from approach_1.src.evaluate import evaluate
from approach_1.src.models import ErrorItem, FindingList, SegmentJudgement
from testing import SCENARIOS

# A1 emits fully-suffixed category names; the scenario catalog uses short names.
_A1_CATEGORY = {
    "incorrect": "incorrect_information",
    "missing": "missing_information",
    "conflict": "conflicting_information",
    "hallucinated": "hallucinated_information",
}

# Relationship names the judge protocol understands, per scenario category.
_RELATIONSHIP = {
    "incorrect": "incorrect",
    "missing": "missing",
    "conflict": "conflict",
    "hallucinated": "hallucination",
}

# Scenarios the A1 *deterministic heuristic* can actually see. Anything outside
# this set (garbled technical words, mid-sentence deletions, semantic-only
# differences) is invisible to the heuristic -- detecting it needs the LLM
# layer, which is exercised separately.
_HEURISTIC_CAPABLE = {"number_change", "negated_fact"}


class _FakeJudge:
    """Canned judge: returns the configured relationship, and re-emits the
    preliminary findings during the global review pass."""

    def __init__(self, relationship: str):
        self.relationship = relationship

    def judge(self, prompt, schema):
        if schema is FindingList:
            import json
            import re

            match = re.search(r"\[.*\]", prompt, flags=re.S)
            if match:
                return schema(findings=[ErrorItem(**item) for item in json.loads(match.group(0))])
            return schema(findings=[])
        return schema(relationship=self.relationship, explanation="fake", severity="high")


def _emitted_categories(report) -> set[str]:
    cats = set()
    for key in ("missing_information", "incorrect_information",
                "conflicting_information", "hallucinated_information"):
        if getattr(report, key):
            cats.add(key)
    return cats


class TestScenarioDetectorHeuristic:
    """Layer 1: what the offline, deterministic evaluator can see by itself."""

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_heuristic_findings_match_scenario(self, scenario):
        report = evaluate(scenario.baseline, scenario.error_side)
        cats = _emitted_categories(report)

        if scenario.name in _HEURISTIC_CAPABLE:
            assert _A1_CATEGORY[scenario.expected_a1_category] in cats
        else:
            # perfect, semantic-only, garbled technical words, and mid-sentence
            # deletions are all invisible to the deterministic heuristic.
            assert cats == set()


class TestScenarioJudgeConsumption:
    """Layer 2: given an LLM verdict, does the pipeline emit the right finding?"""

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_judge_output_is_consumed_correctly(self, scenario):
        if scenario.expected_a1_category is None:
            relationship = "match"  # semantic-only / perfect: LLM says nothing wrong
        else:
            relationship = _RELATIONSHIP[scenario.expected_a1_category]
        judge = _FakeJudge(relationship)

        report = evaluate(scenario.baseline, scenario.error_side, judge=judge)
        cats = _emitted_categories(report)

        if scenario.expected_a1_category is None:
            assert cats == set()
        else:
            assert _A1_CATEGORY[scenario.expected_a1_category] in cats


class TestScenarioAlignment:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_gold_vs_candidate_alignment_is_stable(self, scenario):
        result = align(scenario.baseline, scenario.error_side)
        # every baseline sentence must have a counterpart or be a known miss
        stats = result.stats
        assert stats["gold_segments"] >= 1
        # a mid-sentence deletion (missing_item) stays *inside* a matched pair --
        # that is why the LLM judge must be consulted; it does not surface as
        # an unmatched gold sentence.
        assert stats["gold_segments"] == stats["matched"] + stats["unmatched_gold"] + stats["covered_gold"]
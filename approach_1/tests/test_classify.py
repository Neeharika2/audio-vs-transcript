import json
import re

import pytest

from approach_1.src.align import align
from approach_1.src.classify import CATEGORY_MAP, classify
from approach_1.src.models import ErrorItem, FindingList


class FakeJudge:
    def __init__(self, relationship="match", severity="low"):
        self.relationship = relationship
        self.severity = severity

    def judge(self, prompt, schema):
        if schema is FindingList:
            match = re.search(r"\[.*\]", prompt, flags=re.S)
            if match:
                return schema(findings=[ErrorItem(**item) for item in json.loads(match.group(0))])
            return schema(findings=[])
        return schema(relationship=self.relationship, explanation="fake judgement", severity=self.severity)


GOLD = (
    "The patient takes fifty milligrams of aspirin daily. "
    "He was admitted on Tuesday. "
    "The hospital is on Main Street."
)


class TestClassify:
    def test_perfect_match_no_findings(self):
        result = align(GOLD, GOLD)
        findings, calls = classify(result, FakeJudge(relationship="match"), GOLD, GOLD)
        assert findings == []
        assert calls == 0

    def test_missing_sentence(self):
        candidate = (
            "The patient takes fifty milligrams of aspirin daily. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate)
        findings, calls = classify(result, None, GOLD, candidate)
        cats = [f.category for f in findings]
        assert "missing_information" in cats
        assert any("Tuesday" in (f.reference_text or "") for f in findings)

    def test_hallucinated_sentence(self):
        candidate = (
            "The patient takes fifty milligrams of aspirin daily. "
            "He was admitted on Tuesday. "
            "The hospital is on Main Street. "
            "He also owns a pet dragon."
        )
        result = align(GOLD, candidate)
        findings, calls = classify(result, None, GOLD, candidate)
        cats = [f.category for f in findings]
        assert "hallucinated_information" in cats
        assert any("dragon" in (f.generated_text or "") for f in findings)

    def test_incorrect_pair_from_judge(self):
        candidate = (
            "The patient takes one hundred milligrams of aspirin daily. "
            "He was admitted on Tuesday. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate)
        findings, calls = classify(result, FakeJudge(relationship="incorrect", severity="high"), GOLD, candidate)
        cats = [f.category for f in findings]
        assert "incorrect_information" in cats
        assert calls >= 1

    def test_category_map_covers_all(self):
        assert set(CATEGORY_MAP) == {"missing", "incorrect", "conflict", "hallucination"}

    def test_no_llm_when_judge_none_pairs_skipped(self):
        candidate = GOLD
        result = align(GOLD, candidate)
        findings, calls = classify(result, None, GOLD, candidate)
        assert findings == []
        assert calls == 0

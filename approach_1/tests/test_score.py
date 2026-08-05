import pytest

from approach_1.src.evaluate import evaluate
from approach_1.src.score import error_penalty, score_report
from approach_1.src.models import ErrorItem

GOLD = (
    "The patient takes fifty milligrams of aspirin daily. "
    "He was admitted on Tuesday. "
    "The hospital is on Main Street."
)


class TestScore:
    def test_error_penalty(self):
        findings = [
            ErrorItem(category="missing_information", severity="high"),
            ErrorItem(category="incorrect_information", severity="low"),
        ]
        assert error_penalty(findings) == 12.0

    def test_score_report_perfect(self):
        breakdown, overall, status = score_report([], {"semantic_similarity": 1.0, "entity_recall": 1.0, "wer": 0.0})
        assert overall == 100
        assert status == "Match"

    def test_score_report_high_severity_mismatch(self):
        findings = [ErrorItem(category="conflicting_information", severity="high")]
        _, overall, status = score_report(findings, {"semantic_similarity": 0.99, "entity_recall": 0.95, "wer": 0.0})
        assert status == "Mismatch"
        assert overall <= 90

    def test_score_report_below_threshold(self):
        _, overall, status = score_report([], {"semantic_similarity": 0.8, "entity_recall": 0.8, "wer": 0.2}, threshold=90)
        assert status == "Mismatch"

    def test_no_semantic_redistributes(self):
        breakdown, overall, _ = score_report([], {"entity_recall": 1.0, "wer": 0.0})
        assert overall == 100


class TestEvaluate:
    def test_perfect_report(self):
        report = evaluate(GOLD, GOLD)
        assert report.status == "Match"
        assert report.overall_score == 100
        assert report.alignment.matched == 3
        assert len(report.missing_information) == 0

    def test_missing_detected(self):
        candidate = "The patient takes fifty milligrams of aspirin daily. The hospital is on Main Street."
        report = evaluate(GOLD, candidate)
        assert report.status == "Mismatch"
        assert len(report.missing_information) == 1

    def test_hallucination_detected(self):
        candidate = GOLD + " He owns a pet dragon."
        report = evaluate(GOLD, candidate)
        assert report.status == "Mismatch"
        assert len(report.hallucinated_information) == 1

    def test_signals_present(self):
        report = evaluate(GOLD, GOLD)
        assert report.signals["wer"] == 0.0
        assert report.signals["coverage"] == 1.0
        assert report.meta.llm_calls == 0
        assert report.id.startswith("evl_")

    def test_v2_json_serializable(self):
        import json

        report = evaluate(GOLD, GOLD)
        payload = json.loads(report.model_dump_json())
        assert payload["overall_score"] == 100
        assert payload["alignment"]["matched"] == 3

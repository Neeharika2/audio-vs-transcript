import pytest

from approach_1.src.evaluate import evaluate
from approach_1.src.validation import (
    compute_metrics,
    default_suite,
    evaluate_case,
    run_suite,
)

GOLD_SENTENCES = [
    "The patient takes fifty milligrams of aspirin daily.",
    "He was admitted on Tuesday.",
    "The hospital is on Main Street.",
]


class TestSuite:
    def test_clean_case_no_findings(self):
        from approach_1.src.validation import build_case

        case = build_case(GOLD_SENTENCES, keep=[0, 1, 2])
        findings, annots = evaluate_case(evaluate, case)
        assert annots == []
        assert findings == []

    def test_missing_injection_detected(self):
        from approach_1.src.validation import build_case

        case = build_case(GOLD_SENTENCES, keep=[0, 2])
        findings, annots = evaluate_case(evaluate, case)
        assert any(a.category == "missing_information" for a in annots)
        assert any(f.category == "missing_information" for f in findings)

    def test_hallucination_injection_detected(self):
        from approach_1.src.validation import build_case

        case = build_case(GOLD_SENTENCES, keep=[0, 1, 2], insert=[(2, "He also owns a pet dragon.")])
        findings, annots = evaluate_case(evaluate, case)
        assert any(a.category == "hallucinated_information" for a in annots)
        assert any(f.category == "hallucinated_information" for f in findings)

    def test_metrics_perfect_detector(self):
        # Detector used here is deterministic and should recover all injections.
        results = [evaluate_case(evaluate, case) for case in default_suite()]
        metrics = compute_metrics(results)
        assert metrics["overall"]["recall"] == 1.0
        for category in ("missing_information", "incorrect_information",
                         "conflicting_information", "hallucinated_information"):
            assert metrics[category]["injected"] >= 1

    def test_run_suite_shape(self):
        metrics = run_suite(default_suite())
        assert set(metrics) == {
            "overall",
            "missing_information",
            "incorrect_information",
            "conflicting_information",
            "hallucinated_information",
        }
        assert all("f1" in v for v in metrics.values())

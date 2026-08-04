from approach_1.src.models import EvaluationReport

def calculate_consistency_score(report: EvaluationReport) -> int:
    """Calculate overall score based on the number and severity of errors."""
    # Simple placeholder logic
    total_errors = (
        len(report.missing_information) +
        len(report.incorrect_information) +
        len(report.conflicting_information) +
        len(report.hallucinated_information)
    )
    score = max(0, 100 - (total_errors * 10))
    return score

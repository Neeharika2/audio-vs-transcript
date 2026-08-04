from pydantic import BaseModel
from typing import List, Optional

class ErrorItem(BaseModel):
    category: str  # missing, incorrect, conflicting, hallucinated
    reference_text: Optional[str] = None
    generated_text: Optional[str] = None
    context: Optional[str] = None
    explanation: Optional[str] = None
    severity: str = "medium"  # low, medium, high

class EvaluationReport(BaseModel):
    missing_information: List[ErrorItem] = []
    incorrect_information: List[ErrorItem] = []
    conflicting_information: List[ErrorItem] = []
    hallucinated_information: List[ErrorItem] = []
    overall_score: int = 100
    status: str = "Match"

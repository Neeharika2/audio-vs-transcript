from pydantic import BaseModel, Field
from typing import List, Optional

class SignalEvidence(BaseModel):
    """Deterministic metric values backing a finding.

    Kept as a fixed model (not a free-form dict) so the JSON schema stays
    compatible with Gemini's Developer API, which rejects additionalProperties.
    """
    segment_similarity: Optional[float] = None


class ErrorItem(BaseModel):
    category: str  # missing, incorrect, conflicting, hallucinated
    reference_text: Optional[str] = None
    generated_text: Optional[str] = None
    explanation: Optional[str] = None
    severity: str = "medium"  # low, medium, high
    signal_evidence: Optional[SignalEvidence] = None


class SegmentJudgement(BaseModel):
    relationship: str  # match | incorrect | conflict | missing | hallucination
    explanation: Optional[str] = None
    severity: str = "low"


class FindingList(BaseModel):
    findings: List[ErrorItem] = []


class EvaluationInputs(BaseModel):
    gold_source: Optional[str] = None
    candidate_source: Optional[str] = None
    stt_model: Optional[str] = None
    evaluator: Optional[str] = None


class AlignmentStats(BaseModel):
    gold_segments: int = 0
    candidate_segments: int = 0
    matched: int = 0
    unmatched_gold: int = 0
    unmatched_candidate: int = 0
    covered_gold: int = 0


class ScoreBreakdown(BaseModel):
    semantic: Optional[float] = None
    entity: float = 0.0
    lexical: float = 0.0
    error_penalty: float = 0.0


class Meta(BaseModel):
    llm_calls: int = 0
    latency_ms: int = 0
    generated_at: Optional[str] = None


class EvaluationReportV2(BaseModel):
    id: Optional[str] = None
    inputs: EvaluationInputs = Field(default_factory=EvaluationInputs)
    missing_information: List[ErrorItem] = []
    incorrect_information: List[ErrorItem] = []
    conflicting_information: List[ErrorItem] = []
    hallucinated_information: List[ErrorItem] = []
    overall_score: int = 100
    status: str = "Match"
    alignment: AlignmentStats = Field(default_factory=AlignmentStats)
    signals: dict = Field(default_factory=dict)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    meta: Meta = Field(default_factory=Meta)

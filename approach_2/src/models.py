"""Pydantic models shared across the pipeline."""

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    confidence: float | None = None


class EngineSegment(BaseModel):
    engine: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[Word] = Field(default_factory=list)


class WordOp(BaseModel):
    text: str
    op: str  # "match" | "substitute" | "insert" | "delete"


class AlignedSegment(BaseModel):
    idx: int
    start: float
    end: float
    engine_a: EngineSegment | None = None
    engine_b: EngineSegment | None = None
    agreement: float | None = None
    diff: list[WordOp] = Field(default_factory=list)
    confidence: float = 0.0
    tier: str = "mandatory"  # auto_accept | review_technical | mandatory
    verdict: str = "unreviewed"  # unreviewed | accepted | corrected | rejected
    correction: str | None = None
    llm_judgment: "LLMJudgeVerdict | None" = None


class LLMJudgeVerdict(BaseModel):
    """Structured arbitration from the audio-capable LLM for a disagreement.

    `classification` is one of:
      - "missing"        one engine omitted content that IS in the audio
      - "extra"          a transcript contains content NOT in the audio
      - "hallucinated"   a transcript invented content that is not there
      - "incorrect"      a word/span transcribed wrongly
      - "conflicting"    the two transcripts conflict on something material
      - "accurate"       the LLM listened and the audio matches (false alarm)

    `severity` is one of: "low" | "medium" | "high" | "critical".
    """

    classification: str = "incorrect"
    correct_content: str = ""
    whisper_error: bool = False
    deepgram_error: bool = False
    severity: str = "low"
    explanation: str = ""
    evidence: str = ""


class SpotCheck(BaseModel):
    seed: int
    sample_ids: list[int] = Field(default_factory=list)
    accuracy: float | None = None
    accepted: bool | None = None
    expanded: bool = False
    full_review: bool = False


class ReviewReport(BaseModel):
    audio: str
    engines: list[str]
    segments: list[AlignedSegment]
    spot_check: SpotCheck
    generated_at: str

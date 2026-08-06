from typing import Protocol, TypeVar, cast

from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from pydantic import BaseModel

from approach_1.src.models import EvaluationReport

StructuredT = TypeVar("StructuredT", bound=BaseModel)

class STTModel(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file into raw text."""
        ...

class LLMEvaluator(Protocol):
    def evaluate(self, gold_transcript: str, candidate_transcript: str) -> EvaluationReport:
        """Compare candidate transcript against gold standard transcript."""
        ...

# One model per process; loading Whisper per request would be far too slow.
_whisper_models: dict[str, WhisperModel] = {}

def _get_whisper_model(model_name: str) -> WhisperModel:
    if model_name not in _whisper_models:
        _whisper_models[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_models[model_name]

class WhisperSTT:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    def transcribe(self, audio_path: str) -> str:
        model = _get_whisper_model(self.model_name)
        segments, _info = model.transcribe(audio_path)
        return " ".join(segment.text.strip() for segment in segments)

# Concrete Mock Classes for Testing
class MockSTT:
    def transcribe(self, audio_path: str) -> str:
        return "This is a mock transcript of the audio content."

_EVALUATION_PROMPT = """You are auditing the quality of a machine-generated transcript.

Compare the CANDIDATE transcript against the GOLD (reference) transcript and
categorize every discrepancy into exactly one of these categories:

- missing_information: facts present in the gold transcript but absent from the candidate.
- incorrect_information: facts present in both but stated differently in the candidate (wrong value, wording that changes meaning).
- conflicting_information: facts in the candidate that directly contradict the gold transcript.
- hallucinated_information: facts in the candidate that have no basis in the gold transcript.

Rules:
- One ErrorItem per discrepancy.
- Fill reference_text with the relevant gold passage, generated_text with the candidate passage, and context with the surrounding text from either transcript when helpful.
- severity is "low", "medium", or "high".
- overall_score is 0-100, reflecting how much of the gold content is faithfully reproduced.
- status is "Match" when there are no substantive discrepancies, otherwise "Mismatch".

GOLD transcript:
{gold_transcript}

CANDIDATE transcript:
{candidate_transcript}
"""

class GeminiEvaluator:
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for EVAL_PROVIDER='gemini'")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def evaluate(self, gold_transcript: str, candidate_transcript: str) -> EvaluationReport:
        prompt = _EVALUATION_PROMPT.format(
            gold_transcript=gold_transcript,
            candidate_transcript=candidate_transcript,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationReport,
            ),
        )
        return cast(EvaluationReport, response.parsed)

class GeminiJudge:
    """Segment-level LLM judge for the V2 pipeline (implements Judge protocol)."""

    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for EVAL_PROVIDER='gemini'")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def judge(self, prompt: str, schema: type[StructuredT]) -> StructuredT:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return cast(StructuredT, response.parsed)

class MockEvaluator:
    def evaluate(self, gold_transcript: str, candidate_transcript: str) -> EvaluationReport:
        return EvaluationReport(
            overall_score=95,
            status="Match"
        )

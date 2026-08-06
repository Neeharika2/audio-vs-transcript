import os

from dotenv import load_dotenv

from approach_1.src.evaluator import GeminiJudge, MockSTT, WhisperSTT

load_dotenv()

# Speech-to-Text configuration
# Options: "whisper" (local faster-whisper), "mock"
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper")
STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")

# Evaluator configuration
# Options: "gemini" (Google Gemini API), "mock" (offline heuristics)
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "gemini")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gemini-2.5-flash")

# Semantic embeddings (local sentence-transformers)
# Options: "1"/"true" enabled, anything else disables the embedding signal
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").lower() in ("1", "true", "yes")

# Score threshold: overall_score below this -> status Mismatch
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "90"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_stt_runner():
    if STT_PROVIDER == "whisper":
        return WhisperSTT(model_name=STT_MODEL_NAME)
    if STT_PROVIDER == "mock":
        return MockSTT()
    raise ValueError(f"Unsupported STT_PROVIDER: {STT_PROVIDER!r}")


def get_judge():
    """Return the LLM Judge for the V2 pipeline, or None to run offline heuristics."""
    if EVAL_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for EVAL_PROVIDER='gemini'")
        return GeminiJudge(model_name=EVAL_MODEL_NAME, api_key=GEMINI_API_KEY)
    if EVAL_PROVIDER == "mock":
        return None
    raise ValueError(f"Unsupported EVAL_PROVIDER: {EVAL_PROVIDER!r}")


def get_embedder():
    """Return a lazy sentence embedder when enabled, else None."""
    if not EMBEDDINGS_ENABLED:
        return None
    from approach_1.src.signals import SentenceEmbedder

    return SentenceEmbedder()

import os

from dotenv import load_dotenv

from approach_1.src.evaluator import GeminiJudge, WhisperSTT

load_dotenv()

# Speech-to-Text
STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")

# Evaluator
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gemini-3.5-flash")

# Semantic embeddings (local sentence-transformers)
# Options: "1"/"true" enabled, anything else disables the embedding signal
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Score threshold: overall_score below this -> status Mismatch
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "90"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_stt_runner():
    return WhisperSTT(model_name=STT_MODEL_NAME)


def get_judge():
    if not GEMINI_API_KEY:
        return None
    return GeminiJudge(model_name=EVAL_MODEL_NAME, api_key=GEMINI_API_KEY)


def get_embedder():
    """Return a lazy sentence embedder when enabled, else None."""
    if not EMBEDDINGS_ENABLED:
        return None
    from approach_1.src.signals import SentenceEmbedder

    return SentenceEmbedder()

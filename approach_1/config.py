import os

from dotenv import load_dotenv

from approach_1.src.evaluator import GeminiEvaluator, MockEvaluator, MockSTT, WhisperSTT

load_dotenv()

# Speech-to-Text configuration
# Options: "whisper" (local faster-whisper), "mock"
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper")
STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")

# Evaluator configuration
# Options: "gemini" (Google Gemini API), "mock"
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "gemini")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gemini-2.5-flash")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_stt_runner():
    if STT_PROVIDER == "whisper":
        return WhisperSTT(model_name=STT_MODEL_NAME)
    if STT_PROVIDER == "mock":
        return MockSTT()
    raise ValueError(f"Unsupported STT_PROVIDER: {STT_PROVIDER!r}")


def get_evaluator():
    if EVAL_PROVIDER == "gemini":
        return GeminiEvaluator(model_name=EVAL_MODEL_NAME, api_key=GEMINI_API_KEY)
    if EVAL_PROVIDER == "mock":
        return MockEvaluator()
    raise ValueError(f"Unsupported EVAL_PROVIDER: {EVAL_PROVIDER!r}")

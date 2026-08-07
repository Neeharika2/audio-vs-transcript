import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

DATASET_DIR = BASE_DIR / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"

OUTPUT_DIRS = {
    "whisper": DATASET_DIR / "whisper",
    "deepgram": DATASET_DIR / "deepgram",
}

ENGINE_A = "whisper"
ENGINE_B = "deepgram"

WHISPER_MODEL = "small"

# Deepgram Nova is a cloud API; the key must be provided via the environment.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = "nova-3"

# Evaluation thresholds and sampling (see docs/plan.md).
MATCH_THRESHOLD = 0.55
LOW_CONF_THRESHOLD = 0.6
TIER_AUTO_ACCEPT = 98
TIER_REVIEW = 90
DISAGREE_THRESHOLD = 0.9
SPOT_CHECK_FRACTION = 0.10
SPOT_CHECK_SEED = 42
SPOT_CHECK_ACCEPT = 0.99

# Optional wordlist of domain terms (one per line). Empty disables terminology
# escalation of review_technical segments.
GLOSSARY_PATH = os.environ.get("GLOSSARY_PATH", "")


import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"

OUTPUT_DIRS = {
    "whisper": DATASET_DIR / "whisper",
    "deepgram": DATASET_DIR / "deepgram",
}

WHISPER_MODEL = "base"

# Deepgram Nova is a cloud API; the key must be provided via the environment.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = "nova-2"

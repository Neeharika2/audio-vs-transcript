"""Audio decoding helpers.

Every input file is converted to a 16 kHz mono PCM WAV with ffmpeg so both
engines see identical audio regardless of the original container or codec
(mp4, ogg, ...).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


def to_wav_16k(src: Path, out_dir: Path) -> Path:
    """Decode `src` to a 16 kHz mono PCM WAV in `out_dir`; returns the new path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src), "-vn",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {result.stderr.strip()[-500:]}")
    return out


def extract_span(src: Path, start: float, end: float, out_dir: Path) -> Path:
    """Cut the [start, end] span of `src` to a 16 kHz mono PCM WAV in `out_dir`.

    Used to hand the LLM judge the exact audio clip for a disagreement segment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_{start:.2f}_{end:.2f}.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(src), "-vn",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting span {start}-{end} on {src}: {result.stderr.strip()[-500:]}")
    return out

"""Transcribe every audio file in `dataset/audio` with both STT engines.

For each file both engines transcribe the same 16 kHz mono WAV (decoded with
ffmpeg) and the output is written to:

    dataset/whisper/<name>.txt            full transcript
    dataset/whisper/<name>.segments.json  timestamped segments
    dataset/deepgram/<name>.txt
    dataset/deepgram/<name>.segments.json

Usage:
    python -m approach_2.main
    python -m approach_2.main audio-3.ogg       # transcribe one file only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from approach_2 import config
from approach_2.src.audio import to_wav_16k
from approach_2.src.engines import DeepgramEngine, WhisperEngine


def _audio_files(audio_dir: Path, name: str | None) -> list[Path]:
    if name:
        path = Path(name)
        if not path.is_absolute():
            path = audio_dir / path
        if not path.is_file():
            sys.exit(f"Audio file not found: {path}")
        return [path]
    if not audio_dir.is_dir():
        sys.exit(f"Audio directory not found: {audio_dir}")
    return sorted(p for p in audio_dir.iterdir() if p.is_file())


def _write_outputs(out_dir: Path, stem: str, segments: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = " ".join(s.text for s in segments if s.text)
    (out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / f"{stem}.segments.json").write_text(
        json.dumps([s.model_dump() for s in segments], indent=2),
        encoding="utf-8",
    )


def transcribe_file(audio: Path, engines: list, work_dir: Path) -> None:
    wav = to_wav_16k(audio, work_dir)
    for engine in engines:
        segments = engine.transcribe(wav)
        _write_outputs(config.OUTPUT_DIRS[engine.engine], audio.stem, segments)
        words = sum(len(s.words) for s in segments)
        print(f"  [{engine.engine}] {audio.name}: {len(segments)} segment(s), {words} word(s)")
        print(f"    -> {config.OUTPUT_DIRS[engine.engine] / (audio.stem + '.txt')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="audio file name in dataset/audio (default: all files)")
    args = parser.parse_args()

    engines = [
        WhisperEngine(model_name=config.WHISPER_MODEL),
        DeepgramEngine(api_key=config.DEEPGRAM_API_KEY, model=config.DEEPGRAM_MODEL),
    ]
    files = _audio_files(config.AUDIO_DIR, args.audio)
    print(
        f"Transcribing {len(files)} file(s)\n"
        f"  whisper : {config.WHISPER_MODEL}\n"
        f"  deepgram: {config.DEEPGRAM_MODEL}"
    )

    with tempfile.TemporaryDirectory(prefix="approach2_") as work:
        for audio in files:
            print(f"\n{audio.name}")
            transcribe_file(audio, engines, Path(work))


if __name__ == "__main__":
    main()

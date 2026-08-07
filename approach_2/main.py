"""Approach 2 CLI: transcribe audio with both STT engines, then evaluate it.

`transcribe` writes each engine's output to `dataset/<engine>/`:

    dataset/whisper/<name>.txt            full transcript
    dataset/whisper/<name>.segments.json  timestamped segments
    dataset/deepgram/<name>.txt
    dataset/deepgram/<name>.segments.json

`review` runs the evaluation pipeline (align -> compare -> score -> sample) on
the stored transcripts and writes a report to `dataset/review/<name>/`:

    report.json  report.txt  report.md  report.srt  report.vtt

`judge` runs the audio-grounded LLM judge over disagreement segments and stores
the verdicts on the report (needs `GEMINI_API_KEY`):

    python -m approach_2.main judge                # judge all files
    python -m approach_2.main judge audio-1        # judge one file

Usage:
    python -m approach_2.main transcribe                # all files
    python -m approach_2.main transcribe audio-3.ogg    # one file
    python -m approach_2.main review                    # evaluate all files
    python -m approach_2.main review audio-1            # evaluate one file
    python -m approach_2.main evaluate audio-1          # transcribe (if needed) + evaluate
    python -m approach_2.main judge audio-1             # evaluate + LLM judge disagreements

Interactive review UI (play spans, mark correct/incorrect, correct text):
    uvicorn approach_2.api:app --port 8000
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
            matches = [p for p in audio_dir.iterdir() if p.stem == name and p.is_file()]
            if len(matches) != 1:
                sys.exit(f"Audio file not found: {name}")
            path = matches[0]
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


def _audio_stems(audio_dir: Path, name: str | None) -> list[str]:
    return sorted(p.stem for p in _audio_files(audio_dir, name))


def _transcript_exists(stem: str) -> bool:
    return all((config.OUTPUT_DIRS[engine] / f"{stem}.segments.json").is_file() for engine in config.OUTPUT_DIRS)


def _transcribe_all(name: str | None) -> None:
    engines = [
        WhisperEngine(model_name=config.WHISPER_MODEL),
        DeepgramEngine(api_key=config.DEEPGRAM_API_KEY, model=config.DEEPGRAM_MODEL),
    ]
    files = _audio_files(config.AUDIO_DIR, name)
    print(
        f"Transcribing {len(files)} file(s)\n"
        f"  whisper : {config.WHISPER_MODEL}\n"
        f"  deepgram: {config.DEEPGRAM_MODEL}"
    )

    with tempfile.TemporaryDirectory(prefix="approach2_") as work:
        for audio in files:
            print(f"\n{audio.name}")
            transcribe_file(audio, engines, Path(work))


def _print_summary(report) -> None:
    from collections import Counter

    tiers = Counter(s.tier for s in report.segments)
    sampled = len(report.spot_check.sample_ids)
    print(
        f"  segments: {len(report.segments)}  "
        f"auto_accept={tiers['auto_accept']}  "
        f"review_technical={tiers['review_technical']}  "
        f"mandatory={tiers['mandatory']}  "
        f"review_sample={sampled}"
    )
    accuracy = report.spot_check.accuracy
    if accuracy is None:
        print("  acceptance: no verdicts recorded yet")
    else:
        outcome = (
            "accepted"
            if report.spot_check.accepted
            else "expand sample"
            if report.spot_check.expanded
            else "full review required"
        )
        print(f"  acceptance: accuracy={accuracy:.2%} -> {outcome}")


def _write_report_outputs(report, stem: str) -> None:
    from approach_2.src.export import to_md, to_srt, to_txt, to_vtt

    out_dir = config.DATASET_DIR / "review" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(to_txt(report), encoding="utf-8")
    (out_dir / "report.md").write_text(to_md(report), encoding="utf-8")
    (out_dir / "report.srt").write_text(to_srt(report), encoding="utf-8")
    (out_dir / "report.vtt").write_text(to_vtt(report), encoding="utf-8")
    return out_dir


def _review_all(name: str | None) -> None:
    from approach_2.src.pipeline import load_verdicts, run_pipeline
    from approach_2.src.review import apply_review

    stems = _audio_stems(config.AUDIO_DIR, name)
    print(f"Evaluating {len(stems)} file(s)")
    for stem in stems:
        report = run_pipeline(stem)
        apply_review(report, load_verdicts(stem))
        out_dir = _write_report_outputs(report, stem)
        print(f"\n{stem}")
        _print_summary(report)
        print(f"    -> {out_dir}")


def _judge_all(name: str | None) -> None:
    """Run the LLM judge over disagreement segments and store verdicts."""
    from approach_2.src.judge import GeminiJudge, judge_report
    from approach_2.src.pipeline import load_verdicts, run_pipeline
    from approach_2.src.review import apply_review

    if not config.GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set; add it to the repo-root .env to run the LLM judge")

    stems = _audio_stems(config.AUDIO_DIR, name)
    print(f"LLM-judging {len(stems)} file(s) with {config.GEMINI_MODEL}")
    judge = GeminiJudge()
    for stem in stems:
        report = run_pipeline(stem)
        apply_review(report, load_verdicts(stem))
        audio_path = _audio_files(config.AUDIO_DIR, stem)[0]
        try:
            with tempfile.TemporaryDirectory(prefix="approach2_judge_") as work:
                judge_report(report, audio_path, judge, work_dir=Path(work))
        except Exception as exc:
            print(f"  judge failed for {stem}: {exc}", file=sys.stderr)
        out_dir = _write_report_outputs(report, stem)
        judged = sum(1 for s in report.segments if s.llm_judgment is not None)
        print(f"\n{stem}")
        _print_summary(report)
        print(f"    judged: {judged}/{len(report.segments)} segments")
        print(f"    -> {out_dir}")


def _evaluate(name: str | None) -> None:
    """Transcribe any files missing transcripts, then run the review."""
    files = _audio_files(config.AUDIO_DIR, name)
    missing = [f for f in files if not _transcript_exists(f.stem)]
    if missing:
        engines = [
            WhisperEngine(model_name=config.WHISPER_MODEL),
            DeepgramEngine(api_key=config.DEEPGRAM_API_KEY, model=config.DEEPGRAM_MODEL),
        ]
        print(f"Transcribing {len(missing)} missing file(s)")
        with tempfile.TemporaryDirectory(prefix="approach2_") as work:
            for audio in missing:
                print(f"\n{audio.name}")
                transcribe_file(audio, engines, Path(work))
    _review_all(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transcribe", help="transcribe audio with both engines")
    t.add_argument("audio", nargs="?", help="audio file name in dataset/audio (default: all files)")
    r = sub.add_parser("review", help="evaluate stored transcripts and export a report")
    r.add_argument("audio", nargs="?", help="audio file name (default: all files)")
    e = sub.add_parser("evaluate", help="transcribe missing audio then evaluate it")
    e.add_argument("audio", help="audio file name in dataset/audio")
    j = sub.add_parser("judge", help="run the audio-grounded LLM judge over disagreement segments")
    j.add_argument("audio", nargs="?", help="audio file name (default: all files)")
    args = parser.parse_args()

    if args.command == "evaluate":
        _evaluate(args.audio)
    elif args.command == "review":
        _review_all(args.audio)
    elif args.command == "judge":
        _judge_all(args.audio)
    else:
        _transcribe_all(args.audio)


if __name__ == "__main__":
    main()

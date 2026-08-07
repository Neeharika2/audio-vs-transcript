"""Simple runner: pick an audio + manual transcript, run STT, save the
STT-generated transcript, then print the evaluation findings.

Usage:
    python -m approach_1.compare
    python -m approach_1.compare --audio 3 --gold 3
    python -m approach_1.compare --audio audio-5.mp4 --gold transcript_5.pdf
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from approach_1 import config
from approach_1.src.evaluate import evaluate
from approach_1.src.models import EvaluationInputs

BASE = Path(__file__).resolve().parent
DATASETS = BASE / "datasets"
RECORDINGS = DATASETS / "recordings"
TRANSCRIPTS = DATASETS / "manual_transcripts"
STT_OUT = DATASETS / "stt_generated_transcripts"


def prompt_path(label: str, default: Path) -> Path:
    answer = input(f"{label} [{default}]: ").strip()
    return Path(answer) if answer else default


def resolve_audio(value: str) -> Path:
    if value and value.isdigit():
        matches = list(RECORDINGS.glob(f"audio-{value}.*"))
        if matches:
            return matches[0]
    path = Path(value) if value else RECORDINGS / "audio-3.ogg"
    return path if path.is_absolute() else (BASE / path)


def resolve_gold(value: str) -> Path:
    if value and value.isdigit():
        path = TRANSCRIPTS / f"transcript_{value}.pdf"
        if path.exists():
            return path
    path = Path(value) if value else TRANSCRIPTS / "transcript_3.pdf"
    return path if path.is_absolute() else (BASE / path)


def pdf_to_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit(f"Could not extract text from {path} (is it a scanned PDF?)")
    return result.stdout.strip()


def make_judge():
    if config.DEEPSEEK_API_KEY:
        return config.get_judge()
    return None


def print_findings(report) -> None:
    sections = [
        ("MISSING INFORMATION", report.missing_information),
        ("INCORRECT INFORMATION", report.incorrect_information),
        ("CONFLICTING INFORMATION", report.conflicting_information),
        ("HALLUCINATED INFORMATION", report.hallucinated_information),
    ]
    for name, items in sections:
        print(f"\n== {name} ({len(items)}) ==")
        if not items:
            print("  none")
        for i, f in enumerate(items, 1):
            print(f"  {i}. [{f.severity}] {f.explanation or ''}")
            if f.reference_text:
                print(f"     gold:      {f.reference_text}")
            if f.generated_text:
                print(f"     candidate: {f.generated_text}")

    print("\n== SIGNALS ==")
    for k, v in report.signals.items():
        print(f"  {k}: {v}")
    print(f"\noverall_score: {report.overall_score}")
    print(f"status:        {report.status}")
    print(f"llm_calls:     {report.meta.llm_calls}")


def write_csv(report, path: Path) -> None:
    rows = (
        report.missing_information
        + report.incorrect_information
        + report.conflicting_information
        + report.hallucinated_information
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "severity", "explanation", "reference_text", "generated_text"])
        for item in rows:
            writer.writerow([
                item.category,
                item.severity,
                item.explanation or "",
                item.reference_text or "",
                item.generated_text or "",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", help="audio file path, or dataset index 3/5")
    parser.add_argument("--gold", help="manual transcript path, or dataset index 3/5")
    parser.add_argument("--stt-model", default=config.STT_MODEL_NAME, help="faster-whisper model name")
    parser.add_argument("--force-stt", action="store_true", help="re-transcribe instead of reusing a cached transcript")
    parser.add_argument("--offline", action="store_true", help="skip the LLM judge")
    parser.add_argument("--csv", help="write the findings report to this CSV file")
    args = parser.parse_args()

    audio = resolve_audio(args.audio)
    gold = resolve_gold(args.gold)
    if not audio.exists():
        audio = prompt_path("Audio file", audio)
    if not gold.exists():
        gold = prompt_path("Manual transcript (PDF)", gold)

    print(f"Audio : {audio}")
    print(f"Gold  : {gold}")
    print(f"STT   : faster-whisper ({args.stt_model})")

    from approach_1.src.evaluator import transcribe_cached

    gold_text = pdf_to_text(gold)
    cache_file = STT_OUT / f"{audio.stem}_stt.txt"
    cached = (
        not args.force_stt
        and cache_file.exists()
        and cache_file.stat().st_mtime >= audio.stat().st_mtime
    )
    candidate = transcribe_cached(str(audio), args.stt_model, STT_OUT, force=args.force_stt)
    print(f"STT: {'cached' if cached else 'fresh transcription'} ({len(candidate.split())} words).")

    judge = None if args.offline else make_judge()
    report = evaluate(
        gold_text,
        candidate,
        judge=judge,
        embedder=None,
        inputs=EvaluationInputs(
            gold_source=str(gold),
            candidate_source=str(audio),
            stt_model=args.stt_model,
            evaluator=config.EVAL_MODEL_NAME,
        ),
        threshold=config.SCORE_THRESHOLD,
    )

    print_findings(report)
    if args.csv:
        csv_path = Path(args.csv)
        write_csv(report, csv_path)
        print(f"\nReport written to {csv_path}")


if __name__ == "__main__":
    main()

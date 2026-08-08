"""CLI entry points for the V2 evaluation pipeline.

Usage:
    python -m approach_1.main transcribe <audio_file>          # model -> STT text
    python -m approach_1.main evaluate <audio_file> --gold <gold.txt>
    python -m approach_1.main evaluate-text --gold <gold.txt> --candidate <candidate.txt>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from approach_1 import config
from approach_1.src.models import EvaluationInputs

STT_OUT = Path(__file__).resolve().parent / "datasets" / "stt_generated_transcripts"
TRANSCRIPT_EXTS = {".pdf", ".txt"}


def _load(path: str) -> str:
    """Read a gold/candidate transcript: .txt as text, .pdf via pdftotext."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in TRANSCRIPT_EXTS:
        raise SystemExit(f"Transcript must be a .pdf or .txt file (got '{path}')")
    if suffix == ".pdf":
        result = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            raise SystemExit(f"Could not extract text from the PDF: {path}")
        return result.stdout.strip()
    return p.read_text(encoding="utf-8").strip()


def cmd_transcribe(args) -> None:
    from approach_1.src.evaluator import transcribe_cached

    STT_OUT.mkdir(parents=True, exist_ok=True)
    text = transcribe_cached(args.audio, config.STT_MODEL_NAME, STT_OUT, force=args.force_stt)
    out = STT_OUT / f"{Path(args.audio).stem}_stt.txt"
    print(f"[model {config.STT_MODEL_NAME}] saved: {out}", file=sys.stderr)
    print(text)


def cmd_evaluate(args) -> None:
    judge = config.get_judge()
    embedder = config.get_embedder()

    print(f"STT Model: {config.STT_MODEL_NAME}", file=sys.stderr)
    print(f"Evaluator Model: {config.EVAL_MODEL_NAME}", file=sys.stderr)

    from approach_1.src import runstore
    from approach_1.src.evaluator import transcribe_cached

    candidate = transcribe_cached(args.audio, config.STT_MODEL_NAME, STT_OUT, force=args.force_stt)
    gold = _load(args.gold)
    report = _run(gold, candidate, judge, embedder,
                  inputs=EvaluationInputs(
                      gold_source=args.gold,
                      candidate_source=f"stt:{config.STT_MODEL_NAME}",
                      stt_model=config.STT_MODEL_NAME,
                      evaluator=config.EVAL_MODEL_NAME,
                  ))
    run_id = Path(args.audio).stem
    report.id = run_id
    runstore.save_run(report, run_id, gold, candidate)
    print(f"[saved] /review/runs/{run_id}  (datasets/review/{run_id}.json)", file=sys.stderr)
    print(json.dumps(report.model_dump(), indent=2))


def cmd_evaluate_text(args) -> None:
    gold = _load(args.gold)
    candidate = _load(args.candidate)
    report = _run(gold, candidate, config.get_judge(), config.get_embedder(),
                  inputs=EvaluationInputs(
                      gold_source=args.gold,
                      candidate_source=args.candidate,
                      evaluator=config.EVAL_MODEL_NAME,
                  ))
    print(json.dumps(report.model_dump(), indent=2))


def _run(gold: str, candidate: str, judge, embedder, inputs=None):
    from approach_1.src.evaluate import evaluate

    return evaluate(gold, candidate, judge=judge, embedder=embedder,
                    inputs=inputs, threshold=config.SCORE_THRESHOLD)


def main() -> None:
    parser = argparse.ArgumentParser(prog="approach_1")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="run the STT model on an audio file and save the transcript")
    p_tr.add_argument("audio")
    p_tr.add_argument("--force-stt", action="store_true", help="re-transcribe instead of reusing a cached transcript")
    p_tr.set_defaults(func=cmd_transcribe)

    p_eval = sub.add_parser("evaluate", help="transcribe audio and evaluate against a gold transcript")
    p_eval.add_argument("audio")
    p_eval.add_argument("--gold", required=True)
    p_eval.add_argument("--force-stt", action="store_true", help="re-transcribe instead of reusing a cached transcript")
    p_eval.set_defaults(func=cmd_evaluate)

    p_text = sub.add_parser("evaluate-text", help="compare two transcript files")
    p_text.add_argument("--gold", required=True)
    p_text.add_argument("--candidate", required=True)
    p_text.set_defaults(func=cmd_evaluate_text)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

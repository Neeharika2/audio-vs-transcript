"""CLI entry points for the V2 evaluation pipeline.

Usage:
    python -m approach_1.main evaluate <audio_file> --gold <gold.txt>
    python -m approach_1.main evaluate-text --gold <gold.txt> --candidate <candidate.txt>
    python -m approach_1.main validate
"""

from __future__ import annotations

import argparse
import json
import sys

from approach_1 import config
from approach_1.src.models import EvaluationInputs


def _load(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def cmd_evaluate(args) -> None:
    stt_runner = config.get_stt_runner()
    judge = config.get_judge()
    embedder = config.get_embedder()

    print(f"STT Provider: {config.STT_PROVIDER} ({config.STT_MODEL_NAME})", file=sys.stderr)
    print(f"Evaluator Provider: {config.EVAL_PROVIDER} ({config.EVAL_MODEL_NAME})", file=sys.stderr)

    candidate = stt_runner.transcribe(args.audio)
    gold = _load(args.gold)
    report = _run(gold, candidate, judge, embedder,
                  inputs=EvaluationInputs(
                      gold_source=args.gold,
                      candidate_source=f"stt:{config.STT_MODEL_NAME}",
                      stt_model=config.STT_MODEL_NAME,
                      evaluator=config.EVAL_MODEL_NAME,
                  ))
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


def cmd_validate(args) -> None:
    from approach_1.src.validation import run_suite

    metrics = run_suite()
    print(json.dumps(metrics, indent=2))


def _run(gold: str, candidate: str, judge, embedder, inputs=None):
    from approach_1.src.evaluate import evaluate

    return evaluate(gold, candidate, judge=judge, embedder=embedder,
                    inputs=inputs, threshold=config.SCORE_THRESHOLD)


def main() -> None:
    parser = argparse.ArgumentParser(prog="approach_1")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="transcribe audio and evaluate against a gold transcript")
    p_eval.add_argument("audio")
    p_eval.add_argument("--gold", required=True)
    p_eval.set_defaults(func=cmd_evaluate)

    p_text = sub.add_parser("evaluate-text", help="compare two transcript files")
    p_text.add_argument("--gold", required=True)
    p_text.add_argument("--candidate", required=True)
    p_text.set_defaults(func=cmd_evaluate_text)

    p_val = sub.add_parser("validate", help="run the synthetic validation suite")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

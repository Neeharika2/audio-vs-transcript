"""Offline category-evaluation of both approaches over the shared scenario catalog.

No LLM / API keys required: every number below comes from the deterministic
engines. It covers the five categories — Correct, Missing, Extra, Incorrect,
Conflicting — and saves a report to ``dataset/evaluation_report.json`` while also printing
while also printing the same tables.

Three measurements:
  1. A1_heuristic  Approach 1's offline engine with NO LLM. Deliberately finds
                   only numeric changes and negations; it cannot see mid-sentence
                   deletions (Missing), invented content (Extra) or garbled
                   technical words — those need the LLM layer.
  2. A1_with_llm   The full Approach 1 pipeline, given a correct LLM verdict.
                   Measures whether the pipeline *maps* the discrepancy to the
                   right category.
  3. A2_detector   Approach 2's disagreement detector (algorithm, no LLM).
                   Whether it maps each case to the right A1-style category.

Run:
    python -m testing.eval_accuracy                # both, save default path
    python -m testing.eval_accuracy --approach 1
    python -m testing.eval_accuracy --save path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from testing import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "dataset" / "evaluation_report.json"

CATEGORY_ORDER = [
    ("Correct", "perfect_match"),
    ("Missing", "missing_item"),
    ("Extra", "extra_item"),
    ("Incorrect", "number_change"),
    ("Conflicting", "negated_fact"),
]

_A1_FULL = {
    "incorrect": "incorrect_information",
    "missing": "missing_information",
    "conflict": "conflicting_information",
    "hallucinated": "hallucinated_information",
}
_A1_RELATIONSHIP = {
    "incorrect": "incorrect",
    "missing": "missing",
    "conflict": "conflict",
    "hallucinated": "hallucination",
}


def _category(name: str) -> str:
    for category, scenario in CATEGORY_ORDER:
        if scenario == name:
            return category
    return "Other"


def _a1_expected(scenario) -> str:
    """Suffixed category A1 should emit; 'match' when none is expected."""
    return _A1_FULL.get(scenario.expected_a1_category) or "match"


def _a1_relation(scenario) -> str:
    return _A1_RELATIONSHIP.get(scenario.expected_a1_category) or "match"


def _first_category(report) -> str:
    for key in ("missing_information", "incorrect_information",
                "conflicting_information", "hallucinated_information"):
        if getattr(report, key):
            return key
    return "match"


class _OracleJudge:
    """Replaces the LLM with the verdict a scenario expects (offline only)."""

    def __init__(self, relationship: str):
        self.relationship = relationship

    def judge(self, prompt, schema):
        import re

        from approach_1.src.models import ErrorItem, FindingList, SegmentJudgement

        if schema is FindingList:
            match = re.search(r"\[.*\]", prompt, flags=re.S)
            if match:
                return schema(findings=[ErrorItem(**item) for item in json.loads(match.group(0))])
            return schema(findings=[])
        return SegmentJudgement(
            relationship=self.relationship, explanation="oracle", severity="high"
        )


def _a1_rows(with_judge: bool) -> list[dict]:
    from approach_1.src.evaluate import evaluate

    rows = []
    for s in SCENARIOS:
        report = evaluate(
            s.baseline, s.error_side,
            judge=_OracleJudge(_a1_relation(s)) if with_judge else None,
        )
        rows.append({"name": s.name, "expected": _a1_expected(s), "predicted": _first_category(report)})
    return rows


def _a2_rows() -> list[dict]:
    from approach_2.src.judge import category as a2_category
    from approach_2.src.models import AlignedSegment
    from approach_2.tests.fixtures import seg

    rows = []
    for s in SCENARIOS:
        segment = AlignedSegment(
            idx=0, start=0, end=4,
            engine_a=seg(s.baseline, 0, 4),
            engine_b=seg(s.error_side, 0.1, 3.9, engine="deepgram"),
        )
        rows.append({
            "name": s.name,
            "expected": _a1_expected(s),  # same vocabulary as A1
            "predicted": a2_category(segment),
        })
    return rows


def _score(rows: list[dict]) -> tuple[dict, int, int]:
    """Returns (by_name, ok, total)."""
    by_name, ok, total = {}, 0, 0
    for r in rows:
        total += 1
        match = r["expected"] == r["predicted"]
        ok += match
        by_name[r["name"]] = {"expected": r["expected"], "predicted": r["predicted"], "ok": match}
    return by_name, ok, total


def _pct(ok: int, total: int) -> str:
    return f"{ok}/{total} = {ok / total:.0%}" if total else "0/0"


def _print_table(title: str, by_name: dict) -> None:
    print(title + "\n")
    for s in SCENARIOS:
        r = by_name[s.name]
        print(f"  {s.name:24} expected={r['expected']:24} got={r['predicted']:24} "
              f"{'OK' if r['ok'] else 'MISS'}")
    print()


# ---------------------------------------------------------------------------
# Metric definitions and explanations
# ---------------------------------------------------------------------------

_A1_LABEL = "Approach 1 — offline heuristic engine (no LLM)"
_A1_LLM_LABEL = "Approach 1 — full pipeline + correct LLM verdict"
_A2_LABEL = "Approach 2 — disagreement detector (no LLM)"

METRIC_MEANING = {
    "A1_heuristic": (
        "The deterministic engine with no LLM. It catches numeric changes and "
        "negations only; it is deliberately blind to mid-sentence deletions "
        "(Missing), added content (Extra) and garbled technical words — those "
        "misses are by design and are what the LLM layer is for."
    ),
    "A1_with_llm": (
        "The full Approach 1 pipeline given that the LLM flagged the discrepancy "
        "correctly. Measures the pipeline's mapping: does it place the finding in "
        "the right category (missing / incorrect / conflicting / hallucinated)?"
    ),
    "A2_detector": (
        "The Approach 2 detector (algorithm, no LLM). It speaks the same "
        "category vocabulary as A1 — match, missing_information, "
        "incorrect_information, conflicting_information, hallucinated_information "
        "— so both columns are directly comparable."
    ),
}


def build_report() -> dict:
    m1 = _score(_a1_rows(False))
    m2 = _score(_a1_rows(True))
    m3 = _score(_a2_rows())

    summary = [
        {"metric": "A1_heuristic", "name": _A1_LABEL, "accuracy": _pct(m1[1], m1[2]),
         "meaning": METRIC_MEANING["A1_heuristic"]},
        {"metric": "A1_with_llm", "name": _A1_LLM_LABEL, "accuracy": _pct(m2[1], m2[2]),
         "meaning": METRIC_MEANING["A1_with_llm"]},
        {"metric": "A2_detector", "name": _A2_LABEL, "accuracy": _pct(m3[1], m3[2]),
         "meaning": METRIC_MEANING["A2_detector"]},
    ]

    scenarios = []
    for s in SCENARIOS:
        a1h = m1[0][s.name]
        a1l = m2[0][s.name]
        a2d = m3[0][s.name]
        scenarios.append({
            "category": _category(s.name),
            "scenario": s.name,
            "gold": s.baseline,
            "candidate": s.error_side,
            "expected": {"A1_category": a1h["expected"], "A2_detector": a2d["expected"]},
            "results": {
                "A1_heuristic": {"predicted": a1h["predicted"], "ok": a1h["ok"]},
                "A1_with_llm": {"predicted": a1l["predicted"], "ok": a1l["ok"]},
                "A2_detector": {"predicted": a2d["predicted"], "ok": a2d["ok"]},
            },
        })

    return {
        "dataset": "dataset/test_cases.json (generated from testing/scenarios.py)",
        "scenario_count": len(SCENARIOS),
        "categories_covered": [c for c, _ in CATEGORY_ORDER],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline category accuracy")
    parser.add_argument("--approach", type=int, choices=[1, 2], help="run one approach only")
    parser.add_argument("--save", default=str(DEFAULT_REPORT), help="path for the saved report")
    args = parser.parse_args()

    print("Evaluating the shared scenario catalog (Correct / Missing / Extra / Incorrect / Conflicting)\n")

    if args.approach in (None, 1):
        by, ok, tot = _score(_a1_rows(False))
        print(_A1_LABEL)
        _print_table(_pct(ok, tot), by)
        by, ok, tot = _score(_a1_rows(True))
        print(_A1_LLM_LABEL)
        _print_table(_pct(ok, tot), by)
    if args.approach in (None, 2):
        by, ok, tot = _score(_a2_rows())
        print(_A2_LABEL)
        _print_table(_pct(ok, tot), by)

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    print(f"Report saved -> {path}")
    print("(Real-LLM judge accuracy needs API keys: python -m testing.eval_layer3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
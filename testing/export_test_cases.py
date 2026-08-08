"""Export the shared scenario catalog to a standalone evaluation dataset.

The single source of truth for every injected disagreement used by *both*
approaches' pytest suites lives in :mod:`testing.scenarios` (an error-injection
catalog that deliberately poses known STT failures: number change, negation,
missing word, garbled technical term, etc.). None of it is generated from the
real audio files.

This script exports that catalog unchanged to ``dataset/test_cases.json``, the
submission's evaluation/test-case dataset. Because the JSON is generated straight
from ``testing.scenarios.SCENARIOS``, the submitted dataset can never diverge
from what pytest asserts.

Regenerate whenever the catalog changes:

    python -m testing.export_test_cases

The JSON is **not** production input for ``approach_1.main`` / ``approach_1.api``
(which operate on the real audio + gold transcripts); it is the evaluation
dataset for the requirement "test cases (JSON/CSV) used for evaluation".
"""

from __future__ import annotations

import json
from pathlib import Path

from testing.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]  # repo root
OUT_FILE = ROOT / "dataset" / "test_cases.json"


def scenario_to_dict(s) -> dict:
    """Serialize one scenario, using the catalog's own field names.

    ``baseline`` = gold/reference for Approach 1 and engine A for Approach 2;
    ``error_side`` = candidate for Approach 1 and engine B for Approach 2.
    """
    return {
        "name": s.name,
        "baseline": s.baseline,          # gold (A1) / engine_a (A2)
        "error_side": s.error_side,      # candidate (A1) / engine_b (A2)
        "expected_a1_category": s.expected_a1_category,   # or None = no finding
        "expected_a2_reason": s.expected_a2_reason,      # deterministic A2 flag
        "expected_a2_category": s.expected_a2_category,  # A2 LLM judge bucket
        "semantic_only": s.semantic_only,
    }


def main() -> None:
    data = [scenario_to_dict(s) for s in SCENARIOS]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} scenarios -> {OUT_FILE}")


if __name__ == "__main__":
    main()
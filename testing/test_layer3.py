"""Layer 3 (optional, network + API-gated): real-LLM accuracy over the catalog.

Excluded from the fast offline suite via pytest's ``-m "not layer3"`` addopts.
Run explicitly with:
    pytest -m layer3

The functions call live Gemini / DeepSeek and synthesize audio for Approach 2
(edge-tts), so they require API keys + network. They assert only a sane
accuracy FLOOR over the whole catalog -- never exact equality -- because Layer 3
*evaluates* the judge rather than mocking it (Layers 1 & 2 mock it).
"""

from __future__ import annotations

from collections import Counter

import pytest

from testing.eval_layer3 import (
    ACCURACY_FLOOR,
    evaluate_approach1,
    evaluate_approach2,
)

# Deliberately not using skipif(True), which would skip even under -m layer3.
# Gate on the model/present marker instead: these only run under `-m layer3`.
pytestmark = pytest.mark.layer3


def _passed(results: Counter) -> bool:
    total = results["OK"] + results["MISS"]
    if total == 0:
        return True
    return results["OK"] / total >= ACCURACY_FLOOR


def test_gemini_category_accuracy_floor():
    results = evaluate_approach2()
    assert _passed(results), f"Gemini accuracy below floor {ACCURACY_FLOOR}: {dict(results)}"


def test_deepseek_category_accuracy_floor():
    results = evaluate_approach1()
    assert _passed(results), f"DeepSeek accuracy below floor {ACCURACY_FLOOR}: {dict(results)}"
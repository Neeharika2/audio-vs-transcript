"""Shared test scenarios that exercise both approaches from one source of truth.

A scenario is a known, injected disagreement. It drives:

  * Approach 1 (gold vs candidate STT)  -- ``gold`` / ``candidate``
  * Approach 2 (whisper vs deepgram)    -- ``engine_a`` / ``engine_b``

Neither package depends on the other; both read the same catalog so behaviors
are asserted consistently. Scenarios do NOT encode an LLM's opinion as a hard
assertion -- see Layer 3 notes in the test files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """One injected case with the expected deterministic behavior.

    Field meaning:
      - ``baseline``: the correct source text. Used as ``gold`` (A1) and as
        engine A (A2).
      - ``error_side``: the injected-error variant. Used as ``candidate`` (A1)
        and as engine B (A2).
      - ``expected_a2_reason``: the deterministic ``critical_difference`` reason
        A2 should return, or ``None`` if it should not be flagged.
      - ``expected_a2_category``: the LLM judge category for A2 (lenient -- not
        asserted as an exact equality against a real LLM).
      - ``expected_a1_category``: the finding category A1 should emit
        (``None`` = no finding). Compared against the A1 category vocabulary.
      - ``semantic_only``: the detector *may* flag this, but the deterministic
        layer must NOT turn it into an actual transcription error (used for
        lexical-only / contraction differences).
    """

    name: str
    baseline: str
    error_side: str
    expected_a2_reason: str | None = None
    expected_a2_category: str | None = None
    expected_a1_category: str | None = None
    semantic_only: bool = False


# A single, focused catalog covering the important failure modes. Deliberately
# small -- the point is coverage of failure modes, not raw quantity. Rows whose
# behavior is already exercised deeply in a package's own tests are not repeated
# here (see approach_1/tests/test_classify.py, approach_2/tests/test_judge.py).
SCENARIOS: list[Scenario] = [
    Scenario(
        name="perfect_match",
        baseline="The patient takes twenty milligrams of aspirin daily.",
        error_side="The patient takes twenty milligrams of aspirin daily.",
        expected_a2_reason=None,
        expected_a1_category=None,
    ),
    Scenario(
        name="number_change",
        baseline="The patient takes twenty milligrams of aspirin daily.",
        error_side="The patient takes two hundred milligrams of aspirin daily.",
        expected_a2_reason="number",
        expected_a2_category="incorrect",
        expected_a1_category="incorrect",
    ),
    Scenario(
        name="negated_fact",
        baseline="Anuria requires dialysis.",
        error_side="Anuria does not require dialysis.",
        expected_a2_reason="negation",
        expected_a2_category="incorrect",
        expected_a1_category="conflict",
    ),
    Scenario(
        name="missing_item",
        baseline="This patient requires dialysis twice a week.",
        error_side="This patient requires twice a week.",
        expected_a2_reason="word_added_removed",
        expected_a2_category="missing",
        expected_a1_category="missing",  # A1's heuristic can't see a mid-sentence
        # deletion (token_set_ratio scores it 1.0), but the LLM judge is forced
        # to look at it via classify's word-count guard, so the full pipeline
        # emits "missing".
    ),
    Scenario(
        name="extra_item",
        baseline="The patient takes twenty milligrams of aspirin daily.",
        error_side="The patient takes twenty milligrams of aspirin daily and metformin.",
        expected_a2_reason="word_added_removed",
        expected_a2_category="extra",        # one engine heard extra content
        expected_a1_category="hallucinated", # no basis in the gold transcript
    ),
    Scenario(
        name="technical_word_garbled",
        baseline="The plan is to schedule cholecystectomy.",
        error_side="The plan is to schedule colosyctomy.",
        expected_a2_reason="technical_word",
        expected_a2_category="incorrect",
        expected_a1_category="incorrect",
    ),
    Scenario(
        name="contraction_equivalent",
        baseline="He'd had abdominal pain for two weeks.",
        error_side="He had abdominal pain for two weeks.",
        expected_a2_reason=None,
        expected_a1_category=None,
        semantic_only=True,  # lexical-only; must NOT be treated as an error
    ),
    Scenario(
        name="short_spelling_noise",
        baseline="She lived in seattle for years.",
        error_side="She lived in seattl for years.",
        expected_a2_reason=None,
        expected_a1_category=None,
        semantic_only=True,  # spelling/plural noise is left to the threshold
    ),
]
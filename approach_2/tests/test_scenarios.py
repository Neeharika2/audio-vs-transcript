"""Scenario-driven tests for Approach 2 (two-engine + LLM judge).

Layer 1 (deterministic, no API): the disagreement detector must flag exactly
what a scenario says, for exactly the stated reason.
Layer 2 (fake judge, no API): the pipeline must correctly consume a verdict
returned by the (faked) LLM and attach it to the right segment.

Layer 3 (real Gemini) is deliberately NOT implemented here; it is optional,
API-key gated, and measures aggregate accuracy rather than exact equality.
"""

import pytest

from approach_2.src.judge import critical_difference, judge_report, select_for_judgment
from approach_2.src.models import AlignedSegment, LLMJudgeVerdict, ReviewReport, SpotCheck
from approach_2.tests.fixtures import seg
from testing import SCENARIOS


def _paired(idx, text_a, text_b) -> AlignedSegment:
    a = seg(text_a, 0, 4)
    b = seg(text_b, 0.1, 3.9, engine="deepgram")
    return AlignedSegment(idx=idx, start=0, end=4, engine_a=a, engine_b=b, agreement=1.0)


def _report(segments) -> ReviewReport:
    return ReviewReport(
        audio="scenario",
        engines=["whisper", "deepgram"],
        segments=segments,
        spot_check=SpotCheck(seed=42, sample_ids=[i for i in range(len(segments))]),
        generated_at="2026-01-01T00:00:00Z",
    )


class _FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict
        self.requests = []

    def judge(self, request):
        self.requests.append(request)
        return self.verdict


@pytest.fixture
def fake_audio(tmp_path, monkeypatch):
    """Provide a fake audio file and stub extract_span to write a small WAV."""

    def _stub(path, start, end, out_dir):
        clip = out_dir / "clip.wav"
        clip.write_bytes(b"\x00" * 44)
        return clip

    monkeypatch.setattr("approach_2.src.judge.extract_span", _stub)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00" * 44)
    return audio


class TestScenarioDetector:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_flagging_matches_scenario(self, scenario):
        report = _report([_paired(0, scenario.baseline, scenario.error_side)])
        segment = report.segments[0]
        reason = critical_difference(segment)
        flagged = select_for_judgment(report)

        if scenario.expected_a2_reason is None:
            # perfectly-equal or semantic_only scenarios are not deterministically
            # flagged (semantic meaning is delegated to the LLM layer).
            assert reason is None
            assert flagged == []
        else:
            assert reason == scenario.expected_a2_reason
            assert flagged == [segment]


class TestScenarioJudgeConsumption:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_verdict_attached_to_flagged_segment(self, scenario, fake_audio):
        report = _report([_paired(0, scenario.baseline, scenario.error_side)])
        expected_cls = scenario.expected_a2_category or "incorrect"
        judge = _FakeJudge(
            LLMJudgeVerdict(classification=expected_cls, correct_content=scenario.baseline)
        )
        judge_report(report, fake_audio, judge, work_dir=fake_audio.parent)

        if scenario.expected_a2_reason is None:
            assert judge.requests == []
            assert report.segments[0].llm_judgment is None
        else:
            assert len(judge.requests) == 1
            verdict = report.segments[0].llm_judgment
            assert verdict is not None
            assert verdict.classification == expected_cls
            assert verdict.correct_content == scenario.baseline
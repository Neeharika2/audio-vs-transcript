"""Tests for the audio-grounded LLM judge stage.

No SDK or API key needed: the detector, prompt builder, JSON parsing, and
orchestration are all exercised with a fake judge; `GeminiJudge` is only checked
for its lazy-import + key guards.
"""

from approach_2 import config
from approach_2.src.judge import (
    GeminiJudge,
    JudgeRequest,
    build_prompt,
    critical_difference,
    is_suspicious,
    judge_report,
    parse_verdict,
    select_for_judgment,
)
from approach_2.src.models import AlignedSegment, LLMJudgeVerdict, ReviewReport, SpotCheck
from approach_2.tests.fixtures import seg

import pytest


def _aligned(idx, a_text, b_text, agreement) -> AlignedSegment:
    a = seg(a_text, 0, 4)
    b = seg(b_text, 0.1, 3.9, engine="deepgram")
    return AlignedSegment(idx=idx, start=0, end=4, engine_a=a, engine_b=b, agreement=agreement)


def _report(segments) -> ReviewReport:
    return ReviewReport(
        audio="audio-test",
        engines=["whisper", "deepgram"],
        segments=segments,
        spot_check=SpotCheck(seed=42, sample_ids=[i for i in range(len(segments))]),
        generated_at="2026-01-01T00:00:00Z",
    )


class _FakeJudge:
    """Returns a canned verdict for any request; records audio bytes seen."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.requests = []

    def judge(self, request: JudgeRequest) -> LLMJudgeVerdict | None:
        self.requests.append(request)
        return self.verdict


class TestCriticalDifference:
    def test_negation_flip_flagged(self):
        # "requires" vs "does not require" — a missing "not" changes meaning.
        s = _aligned(0, "patient requires medication", "patient does not require medication", 0.75)
        assert critical_difference(s) == "negation"

    def test_number_change_flagged(self):
        s = _aligned(0, "take 20 milligrams", "take 200 milligrams", 0.6)
        assert critical_difference(s) == "number"

    def test_glossary_term_change_flagged(self, monkeypatch, tmp_path):
        glossary = tmp_path / "terms.txt"
        glossary.write_text("orthotracycline\n", encoding="utf-8")
        from approach_2 import config
        monkeypatch.setattr(config, "GLOSSARY_PATH", str(glossary))
        s = _aligned(0, "on orthoticycline", "on orthotracycline", 0.5)
        assert critical_difference(s) == "glossary_term"

    def test_long_technical_word_substitution_flagged(self):
        # A surgery name garbled — critical even at high agreement.
        s = _aligned(0, "needs cholecystectomy", "needs colosyctomy", 0.97)
        assert critical_difference(s) == "technical_word"

    def test_word_added_or_removed_flagged(self):
        s = _aligned(0, "she used it last summer", "she used it", 0.6)
        assert critical_difference(s) == "word_added_removed"

    def test_short_spelling_noise_ignored(self):
        # Short substitutions are spelling/plural noise, not critical.
        s = _aligned(0, "she lived in seattl", "she lived in seattle", 0.95)
        assert critical_difference(s) is None
        s2 = _aligned(0, "over the counter sprays", "over the counter spray", 0.95)
        assert critical_difference(s2) is None

    def test_article_noise_ignored(self):
        s = _aligned(0, "the patient is here", "a patient is here", 0.9)
        assert critical_difference(s) is None

    def test_critical_signal_overrides_high_agreement(self):
        # High agreement should NOT protect a technical-word substitution.
        s = _aligned(0, "normal intra cardiac", "normal intracardic", 0.9)
        assert critical_difference(s) is not None


class TestDisagreementDetector:
    def test_flags_low_agreement_and_missing_side(self):
        segments = [
            _aligned(0, "hello world", "hello world", 1.0),
            _aligned(1, "dose 20 mg", "dose 200 mg", 0.5),
        ]
        segments[1].engine_b = None
        segments[1].agreement = 0.0
        report = _report(segments)
        assert [s.idx for s in select_for_judgment(report)] == [1]

    def test_high_agreement_bypasses(self):
        report = _report([_aligned(0, "hello world", "hello world", 1.0)])
        assert select_for_judgment(report) == []

    def test_custom_threshold(self):
        report = _report([_aligned(0, "hello world", "hello worled", 0.95)])
        assert select_for_judgment(report, disagree_threshold=0.99) != []
        assert select_for_judgment(report, disagree_threshold=0.90) == []

    def test_is_suspicious_matches(self):
        assert is_suspicious(_aligned(0, "a", "b", 0.3))
        assert not is_suspicious(_aligned(0, "a", "a", 1.0))


class TestBuildPrompt:
    def test_includes_both_transcripts_and_timestamps(self):
        prompt = build_prompt(_aligned(0, "twenty milligrams", "two hundred milligrams", 0.4))
        assert "twenty milligrams" in prompt
        assert "two hundred milligrams" in prompt
        assert "0.00s - 4.00s" in prompt
        assert "Whisper" in prompt and "Deepgram" in prompt

    def test_missing_side_rendered(self):
        s = _aligned(0, "only whisper", "", 0.0)
        s.engine_b = None
        prompt = build_prompt(s)
        assert "only whisper" in prompt
        assert "produced no text" in prompt


class TestParseVerdict:
    def test_parses_plain_json(self):
        text = (
            '{"classification": "incorrect", "correct_content": "20 milligrams", '
            '"whisper_error": true, "deepgram_error": false, "severity": "high", '
            '"explanation": "audio clearly says twenty", "evidence": "twenty"}'
        )
        v = parse_verdict(text)
        assert v is not None
        assert v.classification == "incorrect"
        assert v.correct_content == "20 milligrams"
        assert v.whisper_error is True
        assert v.deepgram_error is False
        assert v.severity == "high"

    def test_parses_fenced_and_noisy(self):
        text = 'Here you go:\n```json\n{"classification": "hallucinated", "severity": "critical", "whisper_error": true, "deepgram_error": true, "explanation": "none of it is in the audio", "evidence": "silence", "correct_content": ""}\n```\n-- end'
        v = parse_verdict(text)
        assert v is not None
        assert v.classification == "hallucinated"
        assert v.severity == "critical"

    def test_rejects_garbage(self):
        assert parse_verdict("") is None
        assert parse_verdict("the model refused to answer") is None
        assert parse_verdict("{not json}") is None

    def test_defaults_fill_missing_fields(self):
        v = parse_verdict('{"classification": "accurate"}')
        assert v is not None
        assert v.severity == "low"
        assert v.explanation == ""


class TestJudgeReport:
    def _patch_extract(self, monkeypatch, tmp_path):
        clip = tmp_path / "span.wav"
        clip.write_bytes(b"RIFFfakewav")
        monkeypatch.setattr(
            "approach_2.src.judge.extract_span",
            lambda src, start, end, out_dir: clip,
        )
        return clip

    def test_only_flagged_segments_get_judged(self, tmp_path, monkeypatch):
        self._patch_extract(monkeypatch, tmp_path)
        segments = [
            _aligned(0, "hello world", "hello world", 1.0),
            _aligned(1, "dose 20 mg", "dose 200 mg", 0.5),
        ]
        report = _report(segments)
        verdict = LLMJudgeVerdict(
            classification="incorrect", correct_content="20 mg",
            whisper_error=True, severity="high", explanation="audio says twenty",
        )
        judge = _FakeJudge(verdict)
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"\x00" * 44)
        judge_report(report, audio, judge, work_dir=tmp_path)

        assert len(judge.requests) == 1
        assert segments[0].llm_judgment is None
        assert segments[1].llm_judgment is not None
        assert segments[1].llm_judgment.classification == "incorrect"

    def test_no_flags_no_calls(self, tmp_path, monkeypatch):
        self._patch_extract(monkeypatch, tmp_path)
        report = _report([_aligned(0, "hello world", "hello world", 1.0)])
        judge = _FakeJudge(LLMJudgeVerdict(classification="accurate"))
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"\x00" * 44)
        judge_report(report, audio, judge, work_dir=tmp_path)
        assert judge.requests == []

    def test_failed_judge_leaves_no_judgment(self, tmp_path, monkeypatch):
        self._patch_extract(monkeypatch, tmp_path)
        report = _report([_aligned(1, "dose 20 mg", "dose 200 mg", 0.5)])
        judge = _FakeJudge(None)
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"\x00" * 44)
        judge_report(report, audio, judge, work_dir=tmp_path)
        assert report.segments[0].llm_judgment is None


class TestGeminiJudge:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            GeminiJudge(api_key="")

    def test_lazy_import_means_construction_needs_no_sdk(self):
        # google-genai is imported lazily inside judge(); constructing the judge
        # with a key must not require the SDK to be installed.
        judge = GeminiJudge(api_key="test-key")
        assert judge.model == "gemini-3.5-flash"

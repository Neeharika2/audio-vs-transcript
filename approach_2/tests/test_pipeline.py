from approach_2.src.export import to_md, to_srt, to_txt, to_vtt
from approach_2.src.models import AlignedSegment, EngineSegment, ReviewReport, SpotCheck
from approach_2.tests.fixtures import seg


def _report() -> ReviewReport:
    a = seg("the patient was alert and oriented", 0, 4)
    b = seg("the patient was alert and oriented", 0.1, 3.9, engine="deepgram")
    aligned = AlignedSegment(
        idx=0, start=0, end=4, engine_a=a, engine_b=b,
        agreement=1.0, confidence=99, tier="auto_accept",
    )
    return ReviewReport(
        audio="audio-test",
        engines=["whisper", "deepgram"],
        segments=[aligned],
        spot_check=SpotCheck(seed=42, sample_ids=[0]),
        generated_at="2026-01-01T00:00:00Z",
    )


class TestExports:
    def test_txt_has_metadata_and_text(self):
        text = to_txt(_report())
        assert "audio-test" in text
        assert "the patient was alert and oriented" in text
        assert "auto_accept" in text

    def test_md_has_table(self):
        text = to_md(_report())
        assert "| # | span | conf | tier | text |" in text

    def test_srt_has_timestamps(self):
        text = to_srt(_report())
        assert "00:00:00,000 --> 00:00:04,000" in text
        assert "the patient was alert and oriented" in text

    def test_vtt_header(self):
        text = to_vtt(_report())
        assert text.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:04.000" in text


class TestPipeline:
    def test_build_report_end_to_end(self):
        from approach_2.src.pipeline import build_report

        a = [seg("first sentence here", 0, 3), seg("second sentence here", 3, 6)]
        b = [seg("first sentence here", 0.1, 2.9, engine="deepgram"), seg("second sentence here", 3.1, 5.9, engine="deepgram")]
        report = build_report(a, b, audio="synthetic")
        assert len(report.segments) == 2
        assert all(0 <= s.confidence <= 100 for s in report.segments)
        assert report.spot_check.seed == 42
        assert all(i < len(report.segments) for i in report.spot_check.sample_ids)

    def test_missing_side_segment_lands_mandatory(self):
        from approach_2.src.pipeline import build_report

        a = [seg("only whisper heard this part", 0, 3)]
        b = []
        report = build_report(a, b, audio="synthetic")
        assert len(report.segments) == 1
        assert report.segments[0].tier == "mandatory"

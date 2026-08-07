"""End-to-end test on the real committed dataset transcripts (plan P9).

Runs the full build_report -> compare -> score -> sample -> apply_review path
against the stored Whisper/Deepgram segments for audio-1. Skipped automatically
if the transcript files are not present (e.g. a fresh checkout).
"""

import pytest

from approach_2 import config
from approach_2.src.export import to_srt, to_txt
from approach_2.src.pipeline import build_report, load_segments
from approach_2.src.review import apply_review


@pytest.fixture(scope="module")
def report():
    has_a = (config.OUTPUT_DIRS[config.ENGINE_A] / "audio-1.segments.json").is_file()
    has_b = (config.OUTPUT_DIRS[config.ENGINE_B] / "audio-1.segments.json").is_file()
    if not (has_a and has_b):
        pytest.skip("dataset transcripts not present")
    segments_a = load_segments(config.ENGINE_A, "audio-1")
    segments_b = load_segments(config.ENGINE_B, "audio-1")
    return build_report(segments_a, segments_b, audio="audio-1")


def test_real_data_produces_report(report):
    assert report.segments
    assert report.spot_check.seed == config.SPOT_CHECK_SEED
    assert report.engines == [config.ENGINE_A, config.ENGINE_B]


def test_all_segments_scored_and_tiered(report):
    assert all(0 <= s.confidence <= 100 for s in report.segments)
    assert all(s.tier in {"auto_accept", "review_technical", "mandatory"} for s in report.segments)
    assert all(s.agreement is not None and 0.0 <= s.agreement <= 1.0 for s in report.segments)


def test_segments_have_at_least_one_engine(report):
    assert all(s.engine_a is not None or s.engine_b is not None for s in report.segments)


def test_sample_ids_valid(report):
    assert all(0 <= i < len(report.segments) for i in report.spot_check.sample_ids)


def test_acceptance_gate_on_real_data(report):
    verdicts = {i: {"verdict": "correct"} for i in report.spot_check.sample_ids}
    reviewed = apply_review(report, verdicts)
    assert reviewed.spot_check.accepted is True


def test_exports_nonempty(report):
    assert "audio-1" in to_txt(report)
    assert to_srt(report).strip()
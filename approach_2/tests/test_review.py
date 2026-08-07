from approach_2.src.compare import compare
from approach_2.src.models import AlignedSegment, ReviewReport, SpotCheck
from approach_2.src.review import (
    acceptance_check,
    apply_review,
    load_glossary,
    sample_review_set,
)
from approach_2.src.score import score
from approach_2.tests.fixtures import seg


def _segments():
    segs = []
    for i, text in enumerate(
        ["one two three", "four five six", "seven eight nine", "ten eleven twelve"]
    ):
        s = AlignedSegment(
            idx=i, start=i * 3.0, end=i * 3.0 + 3.0,
            engine_a=seg(text, i * 3.0, i * 3.0 + 3.0, engine="whisper"),
            engine_b=seg(text, i * 3.0, i * 3.0 + 3.0, engine="deepgram"),
        )
        compare(s)
        score(s)
        segs.append(s)
    return segs


class TestSampling:
    def test_seed_is_reproducible(self):
        sample_a = sample_review_set(_segments(), seed=42)
        sample_b = sample_review_set(_segments(), seed=42)
        assert sample_a == sample_b

    def test_mandatory_segments_always_included(self):
        segs = _segments()
        segs[1].tier = "mandatory"
        sample = sample_review_set(segs, seed=1)
        assert 1 in sample

    def test_disagreement_segments_always_included(self):
        segs = _segments()
        segs[2].agreement = 0.5
        sample = sample_review_set(segs, seed=1)
        assert 2 in sample

    def test_no_duplicate_indices(self):
        segs = _segments()
        for s in segs:
            s.tier = "mandatory"
            s.agreement = 0.5
        sample = sample_review_set(segs, seed=42)
        assert len(sample) == len(set(sample))

    def test_random_fraction_of_rest(self):
        sample = sample_review_set(_segments(), seed=42, fraction=0.5)
        assert len(sample) >= 1


class TestAcceptance:
    def test_high_accuracy_accepted(self):
        assert acceptance_check(1.0) == (True, False)
        assert acceptance_check(0.99) == (True, False)

    def test_mid_accuracy_expands(self):
        assert acceptance_check(0.97) == (False, True)

    def test_low_accuracy_full_review(self):
        assert acceptance_check(0.90) == (False, False)

    def test_none_not_reviewed(self):
        assert acceptance_check(None) == (False, False)


def _report() -> ReviewReport:
    segs = []
    for i, text in enumerate(["one two three", "four five six", "seven eight nine"]):
        s = AlignedSegment(
            idx=i, start=i * 3.0, end=i * 3.0 + 3.0,
            engine_a=seg(text, i * 3.0, i * 3.0 + 3.0, engine="whisper"),
            engine_b=seg(text, i * 3.0, i * 3.0 + 3.0, engine="deepgram"),
        )
        compare(s)
        score(s)
        segs.append(s)
    return ReviewReport(
        audio="t", engines=["whisper", "deepgram"], segments=segs,
        spot_check=SpotCheck(seed=42, sample_ids=[0, 1, 2]),
        generated_at="now",
    )


class TestApplyReview:
    def test_no_verdicts_leaves_acceptance_empty(self):
        report = apply_review(_report(), {})
        assert report.spot_check.accuracy is None
        assert report.spot_check.accepted is None
        assert report.spot_check.full_review is False

    def test_all_correct_accepted(self):
        report = apply_review(_report(), {0: {"verdict": "correct"}, 1: {"verdict": "correct"}, 2: {"verdict": "correct"}})
        assert report.spot_check.accepted is True
        assert report.spot_check.accuracy == 1.0

    def test_one_wrong_triggers_full_review(self):
        report = apply_review(_report(), {0: {"verdict": "correct"}, 1: {"verdict": "incorrect"}, 2: {"verdict": "correct"}})
        assert report.spot_check.accepted is False
        assert report.spot_check.expanded is False
        assert report.spot_check.full_review is True
        assert round(report.spot_check.accuracy, 2) == 0.67

    def test_mid_accuracy_expands(self):
        base = _report()
        base.segments = [
            AlignedSegment(idx=i, start=float(i), end=float(i) + 1, engine_a=seg("one two", float(i), float(i) + 1), engine_b=seg("one two", float(i), float(i) + 1))
            for i in range(20)
        ]
        verdicts = {i: {"verdict": "correct"} for i in range(19)}
        verdicts[19] = {"verdict": "incorrect"}
        report = apply_review(base, verdicts)
        assert report.spot_check.accuracy == 0.95
        assert report.spot_check.accepted is False
        assert report.spot_check.expanded is True

    def test_verdicts_and_corrections_recorded(self):
        report = apply_review(_report(), {1: {"verdict": "incorrect", "correction": "four five nine"}})
        assert report.segments[1].verdict == "incorrect"
        assert report.segments[1].correction == "four five nine"

    def test_correction_only_keeps_unreviewed(self):
        report = apply_review(_report(), {0: {"correction": "fixed"}})
        assert report.segments[0].verdict == "unreviewed"
        assert report.spot_check.accuracy is None


class TestGlossary:
    def test_load_glossary_empty_by_default(self, tmp_path):
        assert load_glossary("") == set()

    def test_load_glossary_parses_lines(self, tmp_path):
        path = tmp_path / "glossary.txt"
        path.write_text("Claritin\nZyrtec\n", encoding="utf-8")
        assert load_glossary(str(path)) == {"claritin", "zyrtec"}

    def test_review_technical_with_term_is_escalated(self, tmp_path, monkeypatch):
        from approach_2 import config

        path = tmp_path / "glossary.txt"
        path.write_text("zyrtec\n", encoding="utf-8")
        monkeypatch.setattr(config, "GLOSSARY_PATH", str(path))
        segs = _report().segments
        segs[1].tier = "review_technical"
        segs[1].engine_a.text = "patient took zyrtec"
        segs[1].engine_b.text = "patient took zyrtec"
        sample = sample_review_set(segs, seed=1)
        assert 1 in sample

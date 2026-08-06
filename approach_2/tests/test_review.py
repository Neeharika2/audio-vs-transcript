from approach_2.src.compare import compare
from approach_2.src.models import AlignedSegment
from approach_2.src.review import acceptance_check, sample_review_set
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

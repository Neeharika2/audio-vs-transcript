from approach_2.src.align import align
from approach_2.tests.fixtures import seg


class TestExactMatch:
    def test_identical_streams_pair_1to1(self):
        a = [seg("the patient was alert", 0, 3), seg("no allergies reported", 3, 6)]
        b = [seg("the patient was alert", 0.1, 2.9), seg("no allergies reported", 3.1, 5.9)]
        result = align(a, b)
        assert len(result) == 2
        assert all(s.engine_a is not None and s.engine_b is not None for s in result)


class TestSplitsAndMerges:
    def test_one_to_three_merge(self):
        a = [seg("the patient had a headache and took aspirin yesterday", 0, 8)]
        b = [
            seg("the patient had a headache", 0, 3),
            seg("and took aspirin", 3, 5),
            seg("yesterday", 5, 8),
        ]
        result = align(a, b)
        assert len(result) == 1
        merged = result[0]
        assert merged.engine_a is not None and merged.engine_b is not None
        assert merged.start == 0
        assert merged.end == 8

    def test_three_to_one_merge(self):
        a = [
            seg("the patient had a headache", 0, 3),
            seg("and took aspirin", 3, 5),
            seg("yesterday", 5, 8),
        ]
        b = [seg("the patient had a headache and took aspirin yesterday", 0, 8)]
        result = align(a, b)
        assert len(result) == 1
        assert result[0].engine_a is not None and result[0].engine_b is not None


class TestDrops:
    def test_dropped_segment_is_missing_side(self):
        a = [seg("first part of the story", 0, 3), seg("second part of the story", 3, 6)]
        b = [seg("first part of the story", 0, 3)]
        result = align(a, b)
        assert len(result) == 2
        matched = next(s for s in result if s.engine_b is not None)
        missing = next(s for s in result if s.engine_b is None)
        assert matched.agreement == 0.0
        assert missing.engine_a is not None
        assert missing.agreement == 0.0


class TestTimestampDrift:
    def test_small_constant_offset_still_aligns(self):
        a = [seg("the quick brown fox jumps over the lazy dog", 5, 15)]
        b = [seg("the quick brown fox jumps over the lazy dog", 5.4, 15.4)]
        result = align(a, b)
        assert len(result) == 1
        assert result[0].engine_a is not None and result[0].engine_b is not None


class TestFillerWords:
    def test_fillers_stripped_for_matching(self):
        a = [seg("um the patient uh felt tired", 0, 4)]
        b = [seg("the patient felt tired", 0, 4)]
        result = align(a, b)
        assert len(result) == 1
        assert result[0].engine_a is not None and result[0].engine_b is not None


class TestUnmatched:
    def test_disjoint_audio_surfaces_as_missing(self):
        a = [seg("something at the start", 0, 2), seg("something at the end", 8, 10)]
        b = [seg("something at the start", 0, 2)]
        result = align(a, b)
        assert len(result) == 2
        assert any(s.engine_b is None for s in result)

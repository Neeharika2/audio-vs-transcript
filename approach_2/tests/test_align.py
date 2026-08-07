from approach_2.src.align import align
from approach_2.src.compare import compare
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

    def test_engine_b_only_span_surfaces_with_missing_a(self):
        a = [seg("something at the start", 0, 2)]
        b = [seg("something at the start", 0, 2), seg("something deepgram alone heard", 8, 10, engine="deepgram")]
        result = align(a, b)
        b_only = next(s for s in result if s.engine_a is None)
        assert b_only.engine_b is not None
        assert b_only.agreement == 0.0


class TestBoundaryDifferences:
    """Regression: one engine splits what the other combines.

    Deepgram's single segment spans three Whisper segments and a trailing
    Deepgram-only "and allegra" tail. The old alignment compared raw segments,
    pulled neighbouring text into each comparison, and dropped the tail as
    "only one engine transcribed this span". The alignment must compare the
    same spoken content instead.
    """

    def test_crossing_split_merge_aligns_same_content(self):
        a = [
            seg("She does have asthma but doesn't require daily medication for this", 37.28, 42.32),
            seg("and does not think it is plaring up.", 42.32, 45.02),
            seg("Her only medication currently is Orthoticycline and Allegra.", 45.58, 49.36),
        ]
        b = [
            seg("she does have asthma", 37.24, 39.16, engine="deepgram"),
            seg(
                "but doesn't require daily medication for this and does not think it is "
                "flaring up her only medication currently is orthotracycline",
                39.40,
                48.42,
                engine="deepgram",
            ),
            seg("and allegra", 48.42, 49.54, engine="deepgram"),
        ]
        result = align(a, b)
        for s in result:
            compare(s)
        assert len(result) == 1
        merged = result[0]
        # Both engines transcribed the whole span; no false "only one engine".
        assert merged.engine_a is not None and merged.engine_b is not None
        # The Deepgram "and allegra" tail is folded into the span, not dropped.
        assert "allegra" in merged.engine_b.text
        # Only the genuine word differences (plaring/flaring, orthoticycline/
        # orthotracycline) count; the boundary difference must not be one.
        assert merged.agreement >= 0.9
        assert [o.op for o in merged.diff].count("match") >= 24

    def test_engine_b_split_tail_is_not_dropped(self):
        a = [seg("Her only medication currently is Orthocycline and Allegra", 45.58, 49.36)]
        b = [
            seg("her only medication currently is orthocycline", 45.58, 48.42, engine="deepgram"),
            seg("and allegra", 48.42, 49.54, engine="deepgram"),
        ]
        result = align(a, b)
        assert len(result) == 1
        merged = result[0]
        assert merged.engine_a is not None and merged.engine_b is not None
        compare(merged)
        assert merged.agreement == 1.0
        assert merged.engine_b.text.endswith("and allegra")

    def test_genuine_word_difference_still_counts(self):
        a = [seg("the patient took aspirin", 0, 3)]
        b = [seg("the patient took ibuprofen", 0, 3, engine="deepgram")]
        result = align(a, b)
        assert len(result) == 1
        compare(result[0])
        assert result[0].agreement == 0.75

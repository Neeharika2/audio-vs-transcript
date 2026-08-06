from approach_2.src.compare import agreement, compare, word_diff
from approach_2.src.models import AlignedSegment
from approach_2.tests.fixtures import seg


class TestWordDiff:
    def test_identical_words_all_match(self):
        ops = word_diff(["a", "b", "c"], ["a", "b", "c"])
        assert [o.op for o in ops] == ["match", "match", "match"]

    def test_substitute(self):
        ops = word_diff(["a", "b"], ["a", "x"])
        assert [o.op for o in ops] == ["match", "substitute"]
        assert ops[1].text == "x"

    def test_insert_and_delete(self):
        ops = word_diff(["a", "b"], ["a", "b", "c"])
        assert [o.op for o in ops] == ["match", "match", "insert"]

        ops = word_diff(["a", "b", "c"], ["a", "b"])
        assert [o.op for o in ops] == ["match", "match", "delete"]

    def test_disjoint_words(self):
        ops = word_diff(["one", "two"], ["three", "four"])
        assert any(o.op != "match" for o in ops)


class TestAgreement:
    def test_perfect_is_one(self):
        assert agreement(word_diff(["a", "b"], ["a", "b"]), 2, 2) == 1.0

    def test_half_mismatch(self):
        diff = word_diff(["a", "b"], ["a", "x"])
        assert agreement(diff, 2, 2) == 0.5

    def test_empty_both_is_one(self):
        assert agreement([], 0, 0) == 1.0


class TestCompare:
    def test_fills_diff_and_agreement(self):
        aligned = AlignedSegment(
            idx=0,
            start=0,
            end=2,
            engine_a=seg("the patient was alert", 0, 2),
            engine_b=seg("the patient was not alert", 0, 2),
        )
        compare(aligned)
        assert aligned.agreement is not None and aligned.agreement < 1.0
        assert any(o.op != "match" for o in aligned.diff)

    def test_missing_side_zero(self):
        aligned = AlignedSegment(idx=0, start=0, end=2, engine_a=seg("text here", 0, 2), engine_b=None)
        compare(aligned)
        assert aligned.agreement == 0.0
        assert aligned.diff == []

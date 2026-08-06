from approach_2.src.compare import compare
from approach_2.src.models import AlignedSegment, Word
from approach_2.src.score import assign_tier, engine_conf, low_conf_ratio, score, segment_confidence
from approach_2.tests.fixtures import seg


def aligned(text_a: str, text_b: str, conf_a: float = 0.98, conf_b: float = 0.98) -> AlignedSegment:
    s = AlignedSegment(
        idx=0,
        start=0,
        end=5,
        engine_a=seg(text_a, 0, 5, engine="whisper", confidence=conf_a),
        engine_b=seg(text_b, 0, 5, engine="deepgram", confidence=conf_b),
    )
    compare(s)
    return s


class TestEngineConf:
    def test_mean_across_both_engines(self):
        s = aligned("a b c", "a b c", conf_a=1.0, conf_b=0.5)
        assert engine_conf(s) == 0.75


class TestLowConfRatio:
    def test_no_low_conf(self):
        s = aligned("a b c", "a b c", conf_a=0.9, conf_b=0.9)
        assert low_conf_ratio(s) == 0.0

    def test_some_low_conf(self):
        low = Word(text="x", confidence=0.3)
        high = Word(text="y", confidence=0.99)
        s = AlignedSegment(
            idx=0, start=0, end=1,
            engine_a=seg("x y", 0, 1, confidence=0.9),
            engine_b=None,
        )
        s.engine_a.words = [low, high]
        assert low_conf_ratio(s) == 0.5


class TestConfidenceFormula:
    def test_clean_segment_scores_high(self):
        s = aligned("the patient was alert and oriented", "the patient was alert and oriented", conf_a=0.98, conf_b=0.98)
        assert segment_confidence(s) >= 98

    def test_missing_side_scores_low(self):
        s = AlignedSegment(idx=0, start=0, end=2, engine_a=seg("only one engine heard this", 0, 2), engine_b=None)
        compare(s)
        assert segment_confidence(s) < 90


class TestTiers:
    def test_bounds(self):
        assert assign_tier(98) == "auto_accept"
        assert assign_tier(99) == "auto_accept"
        assert assign_tier(97) == "review_technical"
        assert assign_tier(90) == "review_technical"
        assert assign_tier(89) == "mandatory"


class TestScore:
    def test_populates_confidence_and_tier(self):
        s = aligned("the patient was alert", "the patient was alert")
        score(s)
        assert s.tier in ("auto_accept", "review_technical", "mandatory")
        assert 0 <= s.confidence <= 100

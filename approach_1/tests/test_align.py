import pytest

from approach_1.src.align import align, segment, segment_sentences, segment_windows


class TestSegmentation:
    def test_sentence_split(self):
        sents = segment_sentences("First sentence. Second one! Third?")
        assert sents == ["First sentence.", "Second one!", "Third?"]

    def test_abbreviation_not_split(self):
        sents = segment_sentences("Dr. Smith lives here. He is a doctor.")
        assert len(sents) == 2

    def test_decimal_not_split(self):
        sents = segment_sentences("The value is 3.5 and it matters.")
        assert len(sents) == 1

    def test_window_fallback(self):
        chunks = segment_windows("one two three four five six seven eight", window=4, overlap=2)
        assert all(len(c.split()) <= 4 for c in chunks)
        assert chunks[0] == "one two three four"

    def test_segment_picks_sentences_or_windows(self):
        segged = segment("A short text without punctuation markers")
        assert len(segged) == 1


GOLD = (
    "The patient takes fifty milligrams of aspirin daily. "
    "He was admitted on Tuesday. "
    "The hospital is on Main Street."
)


class TestAlignment:
    def test_perfect_match(self):
        result = align(GOLD, GOLD)
        assert result.stats["matched"] == 3
        assert result.stats["unmatched_gold"] == 0
        assert result.stats["unmatched_candidate"] == 0

    def test_paraphrase_still_matches(self):
        candidate = (
            "The patient takes 50 mg of aspirin every day. "
            "He was admitted on Tuesday. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate)
        assert result.stats["matched"] == 3
        assert result.stats["unmatched_gold"] == 0

    def test_missing_sentence(self):
        candidate = (
            "The patient takes fifty milligrams of aspirin daily. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate)
        assert result.stats["matched"] == 2
        assert len(result.unmatched_gold) == 1
        assert "Tuesday" in result.unmatched_gold[0].text

    def test_hallucinated_sentence(self):
        candidate = (
            "The patient takes fifty milligrams of aspirin daily. "
            "He was admitted on Tuesday. "
            "The hospital is on Main Street. "
            "The patient also has a pet dragon."
        )
        result = align(GOLD, candidate)
        assert len(result.unmatched_candidate) == 1
        assert "dragon" in result.unmatched_candidate[0].text

    def test_reordered_sentences_align(self):
        candidate = (
            "The hospital is on Main Street. "
            "He was admitted on Tuesday. "
            "The patient takes fifty milligrams of aspirin daily."
        )
        result = align(GOLD, candidate)
        assert result.stats["matched"] == 3
        assert result.stats["unmatched_gold"] == 0

    def test_merged_sentences_covered(self):
        candidate = (
            "The patient takes fifty milligrams of aspirin daily and was admitted on Tuesday. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate)
        assert len(result.covered_gold) >= 1

    def test_empty_inputs(self):
        result = align("", "")
        assert result.stats["matched"] == 0

    def test_mock_embedder_boost(self):
        class DummyEmbedder:
            def similarity(self, a, b):
                return 1.0 if "aspirin" in a and "aspirin" in b else 0.5

        candidate = (
            "The patient takes fifty milligrams of aspirin every single day without fail. "
            "He was admitted on Tuesday. "
            "The hospital is on Main Street."
        )
        result = align(GOLD, candidate, embedder=DummyEmbedder())
        assert result.stats["matched"] == 3

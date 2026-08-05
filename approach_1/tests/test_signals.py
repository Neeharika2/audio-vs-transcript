import pytest

from approach_1.src.signals import (
    char_error_rate,
    compute_signals,
    entity_precision,
    entity_recall,
    extract_entities,
    hallucination_ratio,
    token_coverage,
    word_error_rate,
)


class TestLexicalSignals:
    def test_wer_identical(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_wer_one_substitution(self):
        assert word_error_rate("hello world", "hello there") == pytest.approx(0.5)

    def test_wer_formatting_insensitive(self):
        assert word_error_rate("It costs fifty dollars", "It costs $50") == 0.0

    def test_cer(self):
        assert char_error_rate("cat", "cat") == 0.0
        assert char_error_rate("cat", "catt") == pytest.approx(1 / 3)

    def test_coverage_perfect(self):
        assert token_coverage("a b c", "a b c") == 1.0

    def test_coverage_missing(self):
        assert token_coverage("a b c", "a b") == pytest.approx(2 / 3)

    def test_hallucination_ratio(self):
        assert hallucination_ratio("a b c", "a b x") == pytest.approx(1 / 3)

    def test_empty_gold(self):
        assert token_coverage("", "anything") == 1.0


class TestEntities:
    def test_numbers_extracted(self):
        ents = extract_entities("Take 50 milligrams and 2 tablets")
        assert "50 milligrams" in ents
        assert "2" in ents

    def test_currency_normalized(self):
        assert "50 dollars" in extract_entities("The price is $50")

    def test_time(self):
        ents = extract_entities("Come at 3pm or 15:30")
        assert "15:00" in ents
        assert "15:30" in ents

    def test_dates(self):
        ents = extract_entities("On March 5 and Tuesday")
        assert "march 5" in ents
        assert "tuesday" in ents

    def test_entity_recall(self):
        assert entity_recall("Take 50 milligrams", "Take 50 milligrams") == 1.0
        assert entity_recall("Take 50 milligrams", "Take 25 milligrams") == 0.0

    def test_entity_precision(self):
        assert entity_precision("Take 50 milligrams", "Take 50 milligrams") == 1.0
        assert entity_precision("Take 50 milligrams", "Take 50 milligrams and 3 tablets") == pytest.approx(0.5)

    def test_number_words_vs_digits(self):
        assert entity_recall("Take fifty milligrams", "Take 50 mg") == 1.0


class TestComputeSignals:
    def test_full_signal_dict(self):
        signals = compute_signals("Take fifty milligrams daily", "Take 50 mg daily")
        assert signals["wer"] == 0.0
        assert signals["cer"] == 0.0
        assert signals["coverage"] == 1.0
        assert signals["hallucination_ratio"] == 0.0
        assert signals["entity_recall"] == 1.0
        assert signals["semantic_similarity"] is None

    def test_semantic_with_mock_embedder(self):
        class MockEmbedder:
            def similarity(self, a, b):
                return 0.95

        signals = compute_signals("a b", "a b", embedder=MockEmbedder())
        assert signals["semantic_similarity"] == 0.95

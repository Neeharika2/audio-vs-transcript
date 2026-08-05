import pytest

from approach_1.src.normalize import (
    normalize_number_words,
    normalize_text,
    tokenize,
)


class TestTokenize:
    def test_lowercase_and_punctuation(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_contractions_kept(self):
        assert tokenize("it's a dog") == ["it", "s", "a", "dog"]

    def test_unicode_fullwidth(self):
        assert tokenize("\uff48ello") == ["hello"]


class TestNormalizeText:
    def test_casing_punctuation_whitespace(self):
        assert normalize_text("  The   Stale smell. ") == "the stale smell"

    def test_currency_symbol_before(self):
        assert normalize_text("It costs $50") == "it costs 50 dollars"

    def test_currency_symbol_after(self):
        assert normalize_text("It costs 50$") == "it costs 50 dollars"

    def test_currency_word(self):
        assert normalize_text("It costs USD 50") == "it costs 50 dollars"

    def test_units_joined(self):
        assert normalize_text("5kg of rice") == "5 kilograms of rice"

    def test_units_spaced(self):
        assert normalize_text("5 km away") == "5 kilometers away"

    def test_temperature(self):
        assert normalize_text("36.6°C") == "36 6 degrees celsius"

    def test_temperature_vs_text(self):
        assert normalize_text("36.6°C") == normalize_text("36.6 degrees celsius")

    def test_percent(self):
        assert normalize_text("95% accuracy") == "95 percent accuracy"

    def test_same_meaning_same_normalization(self):
        gold = normalize_text("The medication costs fifty dollars per month.")
        candidate = normalize_text("The medication costs $50 per month.")
        assert gold == candidate


class TestNormalizeNumberWords:
    @pytest.mark.parametrize(
        "words,expected",
        [
            ("fifty", "50"),
            ("twenty three", "23"),
            ("one hundred", "100"),
            ("one hundred and five", "105"),
            ("two thousand twenty four", "2024"),
            ("three million", "3000000"),
            ("twelve", "12"),
            ("ninety nine", "99"),
            ("four hundred fifty", "450"),
        ],
    )
    def test_number_words(self, words, expected):
        assert normalize_number_words(words) == expected

    def test_number_words_in_sentence(self):
        assert normalize_number_words("Take fifty milligrams of aspirin") == "Take 50 milligrams of aspirin"

    def test_mixed_digits_and_words(self):
        assert normalize_number_words("Order 5 and five more") == "Order 5 and 5 more"

    def test_no_false_positives(self):
        assert normalize_number_words("the summer of love") == "the summer of love"

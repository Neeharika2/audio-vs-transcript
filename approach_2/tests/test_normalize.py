"""Number normalization for STT transcripts (digit-by-digit dictation)."""

import pytest

from approach_2.src.normalize import normalize_number_words


@pytest.mark.parametrize(
    "words,expected",
    [
        ("fifty", "50"),
        ("twenty three", "23"),
        ("one hundred", "100"),
        ("three hundred fifty eight", "358"),
        ("two thousand twenty four", "2024"),
        ("twelve", "12"),
        ("ninety nine", "99"),
        ("four hundred fifty", "450"),
    ],
)
def test_canonical_numbers(words, expected):
    assert normalize_number_words(words) == expected


@pytest.mark.parametrize(
    "words,expected",
    [
        ("five nine", "59"),
        ("three fifty eight", "358"),
        ("two sixty", "260"),
        ("one two three", "123"),
        ("nine eleven", "911"),
        ("seventy two", "72"),
    ],
)
def test_digit_speak_numbers(words, expected):
    assert normalize_number_words(words) == expected


def test_digit_speak_in_sentence():
    assert (
        normalize_number_words("he is five nine he has bmi of fifty one")
        == "he is 59 he has bmi of 51"
    )


def test_no_false_positives():
    assert normalize_number_words("the summer of love") == "the summer of love"
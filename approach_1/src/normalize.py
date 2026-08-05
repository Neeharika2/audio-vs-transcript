"""Text normalization for comparing gold vs candidate transcripts.

Produces a canonical form used only for lexical/numeric signals. The LLM
classifier always sees the original, un-normalized text so that meaning is
never lost. Normalization here is deliberately conservative: it fixes
formatting-level differences (casing, punctuation, number spellings, units,
whitespace), not semantic ones.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Tokenization / casing / punctuation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
_MULTI_WS_RE = re.compile(r"\s+")

_CURRENCY_TO_UNIT = {
    "$": "dollars",
    "usd": "dollars",
    "€": "euros",
    "eur": "euros",
    "£": "pounds",
    "gbp": "pounds",
    "¥": "yen",
    "jpy": "yen",
}

_UNIT_TO_WORD = {
    "kg": "kilograms",
    "g": "grams",
    "km": "kilometers",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "mg": "milligrams",
    "ml": "milliliters",
    "l": "liters",
    "°c": "degrees celsius",
    "°f": "degrees fahrenheit",
    "%": "percent",
}


def tokenize(text: str) -> list[str]:
    """Split text into lowercase, punctuation-stripped tokens (NFKD-normalized)."""
    text = unicodedata.normalize("NFKD", text)
    return _MULTI_WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip().split()


def normalize_text(text: str) -> str:
    """Return a canonical whitespace-normalized, lowercased, punctuation-free string."""
    text = unicodedata.normalize("NFKD", text)
    text = _expand_units(text)
    text = _expand_currency(text)
    text = normalize_number_words(text)
    text = _MULTI_WS_RE.sub(" ", text.strip())
    return " ".join(tokenize(text))


# ---------------------------------------------------------------------------
# Units and currency expansion
# ---------------------------------------------------------------------------

def _expand_currency(text: str) -> str:
    """Replace '$50', '€5', '50$' with '50 dollars' etc."""

    def _replace_symbol_first(m: re.Match) -> str:
        unit = _CURRENCY_TO_UNIT[m.group(1).lower()]
        number = m.group(2).replace(",", "")
        return f"{number} {unit}"

    def _replace_number_first(m: re.Match) -> str:
        unit = _CURRENCY_TO_UNIT[m.group(2).lower()]
        number = m.group(1).replace(",", "")
        return f"{number} {unit}"

    text = re.sub(r"([$€£¥])\s*(\d[\d,]*(?:\.\d+)?)", _replace_symbol_first, text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*([$€£¥])(?=\W|$)", _replace_number_first, text)
    text = re.sub(r"\b(usd|eur|gbp|jpy)\b\s*(\d[\d,]*(?:\.\d+)?)", _replace_symbol_first, text, flags=re.IGNORECASE)
    return text


def _expand_units(text: str) -> str:
    """Replace '5kg'/'5 kg' with '5 kilograms' (word boundary aware)."""
    pattern = re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*("
        + "|".join(re.escape(u) for u in _UNIT_TO_WORD)
        + r")(?=\W|$)",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"{m.group(1)} {_UNIT_TO_WORD[m.group(2).lower()]}", text)


# ---------------------------------------------------------------------------
# Number words -> digits
# ---------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

_NUMBER_WORD_ALT = "|".join(sorted(set(_UNITS) | set(_SCALES), key=len, reverse=True))
_NUMBER_WORD_RE = re.compile(
    rf"\b(?:{_NUMBER_WORD_ALT})(?:[\s-]+(?:and|{_NUMBER_WORD_ALT}))*\b",
    re.IGNORECASE,
)


def _words_to_number(words: list[str]) -> int:
    total, current = 0, 0
    for word in words:
        low = word.lower()
        if low in _SCALES:
            current *= _SCALES[low]
            if current >= 1000:
                total += current
                current = 0
        elif low == "and":
            continue
        else:
            current += _UNITS[low]
    return total + current


def normalize_number_words(text: str) -> str:
    """Replace spelled-out numbers with digit strings ('fifty' -> '50')."""

    def _replace(m: re.Match) -> str:
        return str(_words_to_number(m.group(0).replace("-", " ").split()))

    return _NUMBER_WORD_RE.sub(_replace, text)

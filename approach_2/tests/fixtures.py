"""Shared synthetic segment helpers for pipeline tests."""

from approach_2.src.models import EngineSegment, Word


def seg(
    text: str,
    start: float,
    end: float,
    engine: str = "whisper",
    confidence: float = 0.95,
) -> EngineSegment:
    words = [Word(text=w, confidence=confidence) for w in text.split()]
    return EngineSegment(engine=engine, start=start, end=end, text=text, confidence=confidence, words=words)

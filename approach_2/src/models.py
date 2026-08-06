"""Pydantic models shared across the pipeline."""

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    confidence: float | None = None


class EngineSegment(BaseModel):
    engine: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[Word] = Field(default_factory=list)

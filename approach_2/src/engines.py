"""Speech-to-text engines: faster-whisper (local) and Deepgram Nova (cloud).

Both return timestamped `EngineSegment` lists so the later alignment stage can
compare them.
"""

from __future__ import annotations

import math
from pathlib import Path

import requests
from faster_whisper import WhisperModel

from approach_2.src.models import EngineSegment, Word

# One model per process; reloading Whisper per file would be far too slow.
_WHISPER_MODELS: dict[str, WhisperModel] = {}

_DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def _get_whisper_model(model_name: str) -> WhisperModel:
    if model_name not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _WHISPER_MODELS[model_name]


class WhisperEngine:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.engine = "whisper"

    def transcribe(self, audio_path: Path) -> list[EngineSegment]:
        model = _get_whisper_model(self.model_name)
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
        return [
            EngineSegment(
                engine=self.engine,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=math.exp(segment.avg_logprob) if segment.avg_logprob is not None else None,
                words=[Word(text=word.word, confidence=word.probability) for word in (segment.words or [])],
            )
            for segment in segments
            if segment.text.strip()
        ]


class DeepgramEngine:
    def __init__(self, api_key: str, model: str = "nova-3"):
        if not api_key:
            raise ValueError("DeepgramEngine requires an API key (DEEPGRAM_API_KEY)")
        self.api_key = api_key
        self.model = model
        self.engine = "deepgram"

    def transcribe(self, audio_path: Path) -> list[EngineSegment]:
        # Smart formatting / punctuation are disabled so the transcript comes
        # back as plain words, matching Whisper's output for alignment.
        params = {
            "model": self.model,
            "language": "en",
            "smart_format": "false",
            "punctuate": "false",
            "utterances": "true",
            "words": "true",
        }
        response = requests.post(
            _DEEPGRAM_URL,
            params=params,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
            data=audio_path.read_bytes(),
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Deepgram API error {response.status_code}: {response.text[:500]}"
            )
        result = response.json()

        words = result["results"]["channels"][0]["alternatives"][0].get("words") or []
        utterances = result["results"].get("utterances") or []
        if utterances:
            return [_segment_from_utterance(u) for u in utterances if (u.get("transcript") or "").strip()]
        if not words:
            return []
        start = min(w.get("start", 0.0) for w in words)
        end = max(w.get("end", start) for w in words)
        text = " ".join(w["word"] for w in words)
        return [
            EngineSegment(
                engine=self.engine,
                start=start,
                end=end,
                text=text,
                confidence=_mean_conf(words),
                words=[Word(text=w["word"], confidence=w.get("confidence")) for w in words],
            )
        ]


def _segment_from_utterance(u: dict) -> EngineSegment:
    words = u.get("words") or []
    return EngineSegment(
        engine="deepgram",
        start=u.get("start", 0.0),
        end=u.get("end", u.get("start", 0.0)),
        text=(u.get("transcript") or "").strip(),
        confidence=u.get("confidence"),
        words=[Word(text=w["word"], confidence=w.get("confidence")) for w in words],
    )


def _mean_conf(words: list[dict]) -> float | None:
    confs = [w["confidence"] for w in words if "confidence" in w]
    return round(sum(confs) / len(confs), 4) if confs else None

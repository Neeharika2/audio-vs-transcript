import json
from pathlib import Path
from typing import TypeVar, cast

from faster_whisper import WhisperModel
from openai import OpenAI
from pydantic import BaseModel

StructuredT = TypeVar("StructuredT", bound=BaseModel)

# One model per process; loading Whisper per request would be far too slow.
_whisper_models: dict[str, WhisperModel] = {}


def _get_whisper_model(model_name: str) -> WhisperModel:
    if model_name not in _whisper_models:
        _whisper_models[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_models[model_name]


class WhisperSTT:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    def transcribe(self, audio_path: str) -> str:
        model = _get_whisper_model(self.model_name)
        segments, _info = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments)


def transcribe_cached(
    audio_path: str,
    model_name: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> str:
    """Transcribe audio, reusing a cached transcript when the audio is unchanged.

    The cache file is `<cache_dir>/<audio_stem>_stt.txt`. It is reused when it
    already exists and is newer than the audio file; pass `force=True` to
    re-transcribe regardless.
    """
    audio = Path(audio_path)
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{audio.stem}_stt.txt"
        if not force and cache.exists() and cache.stat().st_mtime >= audio.stat().st_mtime:
            return cache.read_text(encoding="utf-8")
        text = WhisperSTT(model_name=model_name).transcribe(str(audio))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
        return text
    return WhisperSTT(model_name=model_name).transcribe(str(audio))


class DeepSeekJudge:
    """OpenAI-compatible DeepSeek judge (implements the Judge protocol)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required to run the DeepSeek judge")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def judge(self, prompt: str, schema: type[StructuredT]) -> StructuredT:
        expected = json.dumps(schema.model_json_schema(), indent=2)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Return only a single valid JSON object that matches this schema:\n" + expected,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return cast(StructuredT, schema.model_validate(json.loads(content)))

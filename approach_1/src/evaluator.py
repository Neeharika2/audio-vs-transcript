import json
import re
import time
from pathlib import Path
from typing import TypeVar, cast

import httpx
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


def _extract_json_object(text: str) -> dict:
    """Pull the JSON object out of a model reply that may wrap it in prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model reply")
    return json.loads(text[start : end + 1])


def _rate_limit_delay(exc: Exception) -> float | None:
    """Extract the suggested retry delay from a Gemini 429 error, if present."""
    text = str(exc)
    m = re.search(r"Please retry in ([\d.]+)s", text)
    if m:
        return float(m.group(1))
    m = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', text)
    if m:
        return float(m.group(1))
    return None


class GeminiJudge:
    """Gemini structured-output judge (implements the Judge protocol)."""

    _MAX_RETRIES = 3
    _RETRY_DELAY = 2.0

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to run the Gemini judge")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            # genai's default internal transport can hang on the TLS handshake
            # in some environments, so inject a plain httpx client instead.
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={
                    "httpx_client": httpx.Client(
                        timeout=self.timeout,
                        follow_redirects=True,
                    )
                },
            )
        return self._client

    def judge(self, prompt: str, schema: type[StructuredT]) -> StructuredT:
        """Ask Gemini for structured output matching `schema` and validate it."""
        from google.genai import types

        client = self._get_client()
        message = (
            "Return only a single valid JSON object that matches this schema:\n"
            + json.dumps(schema.model_json_schema(), indent=2)
            + "\n\n"
            + prompt
        )
        last_error: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=message,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("Gemini judge returned an empty response")
                try:
                    return cast(StructuredT, schema.model_validate_json(text))
                except Exception:
                    return cast(StructuredT, schema.model_validate(_extract_json_object(text)))
            except Exception as exc:  # transient API/rate errors; retry, then raise
                last_error = exc
                delay = _rate_limit_delay(exc)
                if delay is None:
                    delay = self._RETRY_DELAY * (attempt + 1)
                else:
                    # The API tells us exactly how long to wait on a quota hit;
                    # honour it so a parallel burst can drain the free-tier
                    # 5 RPM limit instead of colliding on instant retries.
                    delay = min(delay + 1.0, 60.0)
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"Gemini judge failed after {self._MAX_RETRIES} attempts: {last_error}"
        )

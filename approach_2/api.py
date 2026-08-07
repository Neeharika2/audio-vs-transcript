"""FastAPI review server: serve audio + reports, accept reviewer verdicts.

The static page (`static/review.html`) plays the audio span for each segment,
shows the word diff, and posts verdicts/corrections back here. Verdicts persist
to a JSON sidecar next to the report; acceptance is recomputed on every load.

Run with:
    uvicorn approach_2.api:app --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from approach_2 import config
from approach_2.src.pipeline import load_verdicts, run_pipeline, save_verdicts
from approach_2.src.review import apply_review

app = FastAPI(title="Approach 2 review")


class ReviewBody(BaseModel):
    idx: int
    verdict: str | None = None
    correction: str | None = None


def _audio_file(stem: str) -> Path:
    if not config.AUDIO_DIR.is_dir():
        raise HTTPException(404, "no audio directory")
    for path in config.AUDIO_DIR.iterdir():
        if path.stem == stem and path.is_file():
            return path
    raise HTTPException(404, f"no audio for {stem}")


def _report(stem: str):
    try:
        report = run_pipeline(stem)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no stored transcripts for {stem}; run transcribe first") from exc
    return apply_review(report, load_verdicts(stem))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.BASE_DIR / "static" / "review.html")


@app.get("/files")
def list_files() -> list[str]:
    if not config.AUDIO_DIR.is_dir():
        return []
    return [p.stem for p in sorted(config.AUDIO_DIR.iterdir()) if p.is_file()]


@app.get("/audio/{stem}")
def audio(stem: str) -> FileResponse:
    return FileResponse(_audio_file(stem))


@app.get("/report/{stem}")
def report(stem: str) -> dict:
    return _report(stem).model_dump(mode="json")


@app.post("/review/{stem}")
def review(stem: str, body: ReviewBody) -> dict:
    verdicts = load_verdicts(stem)
    entry = verdicts.setdefault(body.idx, {})
    if body.verdict is not None:
        if body.verdict:
            entry["verdict"] = body.verdict
        else:
            entry.pop("verdict", None)
    if body.correction is not None:
        entry["correction"] = body.correction or None
    save_verdicts(stem, verdicts)
    return _report(stem).model_dump(mode="json")

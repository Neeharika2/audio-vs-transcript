"""FastAPI review server: serve audio + reports, accept reviewer verdicts.

The static page (`static/review.html`) plays the audio span for each segment,
shows the word diff, and posts verdicts/corrections back here. Verdicts persist
to a JSON sidecar next to the report; acceptance is recomputed on every load.

Run with:
    uvicorn approach_2.api:app --port 8000
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from approach_2 import config
from approach_2.src.judge import category as detector_category
from approach_2.src.judge import critical_difference, select_for_judgment
from approach_2.src.models import AlignedSegment, EngineSegment
from approach_2.src.pipeline import apply_judgments, load_judgments, load_verdicts, run_pipeline, save_verdicts
from approach_2.src.review import apply_review

app = FastAPI(title="Approach 2 review")

_SHARED_DATASET = config.BASE_DIR.parent / "dataset" / "test_cases.json"

# A1-style category emitted by the deterministic detector, per the shared catalog.
_A1_CATEGORY_MAP = {
    "incorrect": "incorrect_information",
    "missing": "missing_information",
    "conflict": "conflicting_information",
    "hallucinated": "hallucinated_information",
}

CATEGORY_MATCH = "match"


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
    apply_review(report, load_verdicts(stem))
    apply_judgments(report, load_judgments(stem))
    return report


def _report_payload(stem: str) -> dict:
    """Report dict with each segment tagged for which LLM judged.

    `selected_for_llm` is True when the deterministic detector would send the
    segment to the judge (critical signal, missing side, or agreement below
    `LLM_DISAGREE_THRESHOLD`). A segment that is not selected and has no stored
    verdict was auto-accepted — never sent to the LLM, so it must not read as
    "unjudged".
    """
    report = _report(stem)
    selected = {s.idx for s in select_for_judgment(report)}
    payload = report.model_dump(mode="json")
    for seg in payload["segments"]:
        seg["llm_selected"] = seg["idx"] in selected
    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.BASE_DIR / "static" / "review.html")


@app.get("/new")
def new_page() -> FileResponse:
    return FileResponse(config.BASE_DIR / "static" / "new.html")


@app.post("/evaluate")
def evaluate_new(
    audio: UploadFile = File(...),
) -> dict:
    """Run a new evaluation: save the audio, transcribe it with both engines,
    and create the stored report so it appears on the main audit page."""
    dest = config.AUDIO_DIR / (audio.filename or "audio.wav")
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(audio.file.read())

    from approach_2.src.audio import to_wav_16k
    from approach_2.src.engines import DeepgramEngine, WhisperEngine

    stem = dest.stem
    engines = [
        WhisperEngine(model_name=config.WHISPER_MODEL),
        DeepgramEngine(api_key=config.DEEPGRAM_API_KEY, model=config.DEEPGRAM_MODEL),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="approach2_") as work:
            wav = to_wav_16k(dest, Path(work))
            for engine in engines:
                segments = engine.transcribe(wav)
                out_dir = config.OUTPUT_DIRS[engine.engine]
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{stem}.txt").write_text(
                    " ".join(s.text for s in segments if s.text) + "\n",
                    encoding="utf-8",
                )
                (out_dir / f"{stem}.segments.json").write_text(
                    json.dumps([s.model_dump() for s in segments], indent=2),
                    encoding="utf-8",
                )
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {exc}")

    report = run_pipeline(stem)
    apply_review(report, load_verdicts(stem))
    report_dir = config.DATASET_DIR / "review" / stem
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"stem": stem, "segments": len(report.segments)}


@app.get("/framework")
def framework_page() -> FileResponse:
    return FileResponse(config.BASE_DIR / "static" / "framework.html")


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
    return _report_payload(stem)


@app.get("/validate")
def validate() -> dict:
    """Run the shared synthetic catalog through A2's deterministic detector.

    Each test case is a known injected disagreement (from
    `dataset/test_cases.json`, generated from `testing/scenarios.py`).
    Approach 2 consumes the same catalog file as Approach 1 but evaluates every
    case through its own detector, so the two results columns are comparable.
    """
    if not _SHARED_DATASET.is_file():
        raise HTTPException(404, f"no shared synthetic dataset at {_SHARED_DATASET}")
    cases = json.loads(_SHARED_DATASET.read_text(encoding="utf-8"))

    results = []
    passed = 0
    for i, case in enumerate(cases):
        name = case["name"]
        seg = AlignedSegment(
            idx=i,
            start=0,
            end=4,
            engine_a=EngineSegment(engine=config.ENGINE_A, start=0, end=4, text=case["baseline"]),
            engine_b=EngineSegment(engine=config.ENGINE_B, start=0.1, end=3.9, text=case["error_side"]),
        )
        expected_raw = case.get("expected_a1_category")
        expected = _A1_CATEGORY_MAP.get(expected_raw) if expected_raw else CATEGORY_MATCH
        predicted = detector_category(seg)
        signal = critical_difference(seg)
        ok = predicted == expected
        passed += ok
        results.append({
            "name": name,
            "expected": expected,
            "predicted": predicted,
            "signal": signal or "none",
            "passed": ok,
            "whisper": case["baseline"],
            "deepgram": case["error_side"],
            "analysis": _a2_analysis(name, expected, signal, ok),
        })
    return {"total": len(cases), "passed": passed, "results": results}


def _a2_analysis(name: str, expected: str, signal: str | None, ok: bool) -> str:
    if expected == "match":
        return "No discrepancy expected. The detector found no critical signal."
    if ok:
        return f"Detected via the `{signal}` signal and mapped to the right category."
    return f"Expected {expected} but the detector reported something else (signal `{signal}`)."


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
    if not entry:
        verdicts.pop(body.idx, None)
    save_verdicts(stem, verdicts)
    return _report_payload(stem)

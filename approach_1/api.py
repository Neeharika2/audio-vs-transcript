import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from approach_1 import config
from approach_1.src.evaluate import evaluate
from approach_1.src.models import EvaluationInputs, EvaluationReportV2

app = FastAPI(title="Audio vs Transcript Evaluator")

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPT_EXTS = {".pdf", ".txt"}
STT_OUT = BASE_DIR / "datasets" / "stt_generated_transcripts"
SHARED_DATASET = BASE_DIR.parent / "dataset" / "test_cases.json"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Single-page review UI: click Run, upload audio + manual transcript."""
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/framework", response_class=HTMLResponse)
def framework() -> str:
    """Framework validation page: shared synthetic test cases + analysis."""
    return (BASE_DIR / "static" / "framework.html").read_text(encoding="utf-8")


def _extract_text(upload: UploadFile) -> str:
    """Extract the reference text from an uploaded manual transcript (PDF or TXT)."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in TRANSCRIPT_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Manual transcript must be a .pdf or .txt file (got {suffix or 'none'})",
        )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        path = tmp.name
    try:
        if suffix == ".pdf":
            result = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from the uploaded PDF")
            return result.stdout.strip()
        return Path(path).read_text(encoding="utf-8").strip()
    finally:
        os.unlink(path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationReportV2)
def evaluate_endpoint(
    audio: UploadFile = File(...),
    manual_transcript: UploadFile = File(None),
    gold_transcript: str = Form(None),
) -> EvaluationReportV2:
    """Compare an audio file against a manual (gold) transcript.

    The manual transcript is the reference; `gold_transcript` is accepted as a
    paste-in alias of the same thing for backward compatibility.
    """
    if manual_transcript is not None:
        gold = _extract_text(manual_transcript)
    elif gold_transcript:
        gold = gold_transcript
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a manual transcript file (.pdf or .txt) or paste it as gold_transcript",
        )

    stt_runner = config.get_stt_runner()
    judge = config.get_judge()
    embedder = config.get_embedder()

    audio_bytes = audio.file.read()
    digest = hashlib.sha256(audio_bytes).hexdigest()[:12]
    cache_path = STT_OUT / f"{Path(audio.filename or 'audio').stem}_{digest}_stt.txt"
    if cache_path.exists():
        candidate_transcript = cache_path.read_text(encoding="utf-8")
    else:
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        if audio_path is None:
            raise RuntimeError("Failed to create temporary audio file")
        try:
            candidate_transcript = stt_runner.transcribe(audio_path)
        finally:
            os.unlink(audio_path)
        STT_OUT.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(candidate_transcript, encoding="utf-8")

    return evaluate(
        gold,
        candidate_transcript,
        judge=judge,
        embedder=embedder,
        inputs=EvaluationInputs(
            gold_source="user",
            candidate_source=f"stt:{config.STT_MODEL_NAME}",
            stt_model=config.STT_MODEL_NAME,
            evaluator=config.EVAL_MODEL_NAME,
        ),
        threshold=config.SCORE_THRESHOLD,
    )


@app.post("/evaluate-text", response_model=EvaluationReportV2)
def evaluate_text(
    gold_transcript: str = Form(...),
    candidate_transcript: str = Form(...),
) -> EvaluationReportV2:
    """Compare two texts directly (no audio/STT)."""
    return evaluate(
        gold_transcript,
        candidate_transcript,
        judge=config.get_judge(),
        embedder=config.get_embedder(),
        inputs=EvaluationInputs(
            gold_source="user",
            candidate_source="user",
            evaluator=config.EVAL_MODEL_NAME,
        ),
        threshold=config.SCORE_THRESHOLD,
    )


# -----------------------------------------------------------------------------
# Framework validation over the shared synthetic test catalog
# -----------------------------------------------------------------------------

_A1_CATEGORY = {
    "incorrect": "incorrect_information",
    "missing": "missing_information",
    "conflict": "conflicting_information",
    "hallucinated": "hallucinated_information",
}

_RELATIONSHIP = {
    "incorrect": "incorrect",
    "missing": "missing",
    "conflict": "conflict",
    "hallucinated": "hallucination",
}

# Scenarios the A1 *deterministic heuristic* can actually see without an LLM.
_HEURISTIC_CAPABLE = {"number_change", "negated_fact"}

_FINDING_KEYS = (
    "missing_information",
    "incorrect_information",
    "conflicting_information",
    "hallucinated_information",
)


class _OracleJudge:
    """Emulates a correct LLM: returns the scenario's expected relationship."""

    def __init__(self, relationship: str):
        self.relationship = relationship

    def judge(self, prompt, schema):
        import re

        from approach_1.src.models import ErrorItem, FindingList, SegmentJudgement

        if schema is FindingList:
            match = re.search(r"\[.*\]", prompt, flags=re.S)
            if match:
                return schema(findings=[ErrorItem(**item) for item in json.loads(match.group(0))])
            return schema(findings=[])
        return SegmentJudgement(
            relationship=self.relationship, explanation="oracle", severity="high"
        )


def _emitted_categories(report) -> list[str]:
    return [key for key in _FINDING_KEYS if getattr(report, key)]


def _a1_analysis(name: str, expected: str, heuristic_ok: bool) -> str:
    if expected == "match":
        return "No discrepancy expected. The heuristic correctly leaves the case alone."
    if heuristic_ok:
        return "Detected by the deterministic heuristic (entity / negation signal)."
    if name in _HEURISTIC_CAPABLE:
        return "Expected, but the heuristic missed it (no LLM configured); the pipeline with the judge catches it."
    return "Invisible to the heuristic — a deep diff (deletion / added content / garbled word) that only the LLM layer can classify."


@app.get("/validate")
def validate() -> dict:
    """Run the shared synthetic test catalog through Approach 1.

    Each case is compared gold->candidate twice:
      - `heuristic`: the deterministic evaluator with no LLM (what A1 can see
        offline alone),
      - `pipeline`: the full approach_1 pipeline with a correct LLM verdict.
    Expected category comes from the same `dataset/test_cases.json` both
    approaches share, so the A1 and A2 columns are comparable.
    """
    if not SHARED_DATASET.is_file():
        raise HTTPException(404, f"no shared synthetic dataset at {SHARED_DATASET}")
    cases = json.loads(SHARED_DATASET.read_text(encoding="utf-8"))

    results = []
    passed_total = 0
    for case in cases:
        name = case["name"]
        gold = case["baseline"]
        candidate = case["error_side"]
        raw = case.get("expected_a1_category")
        expected = _A1_CATEGORY.get(raw) if raw else "match"

        heuristic_report = evaluate(gold, candidate)
        heuristic_cats = _emitted_categories(heuristic_report)

        if raw:
            judge = _OracleJudge(_RELATIONSHIP[raw])
            pipeline_cats = _emitted_categories(evaluate(gold, candidate, judge=judge))
        else:
            pipeline_cats = heuristic_cats

        heuristic_ok = (not heuristic_cats) if expected == "match" else expected in heuristic_cats
        pipeline_ok = (not pipeline_cats) if expected == "match" else expected in pipeline_cats
        passed = pipeline_ok
        passed_total += passed

        results.append({
            "name": name,
            "gold": gold,
            "candidate": candidate,
            "expected": expected,
            "heuristic_detected": sorted(heuristic_cats) or ["match"],
            "heuristic_passed": heuristic_ok,
            "pipeline_detected": sorted(pipeline_cats) or ["match"],
            "pipeline_passed": pipeline_ok,
            "passed": passed,
            "analysis": _a1_analysis(name, expected, heuristic_ok),
        })
    return {"total": len(cases), "passed": passed_total, "results": results}

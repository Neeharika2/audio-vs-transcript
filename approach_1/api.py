import hashlib
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Single-page review UI: click Run, upload audio + manual transcript."""
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


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

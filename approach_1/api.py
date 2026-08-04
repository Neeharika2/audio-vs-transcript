import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from approach_1 import config
from approach_1.src.models import EvaluationReport

app = FastAPI(title="Audio vs Transcript Evaluator")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationReport)
def evaluate(audio: UploadFile = File(...), gold_transcript: str = Form(...)) -> EvaluationReport:
    stt_runner = config.get_stt_runner()
    evaluator = config.get_evaluator()

    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio.file.read())
        audio_path = tmp.name

    try:
        candidate_transcript = stt_runner.transcribe(audio_path)
        return evaluator.evaluate(gold_transcript, candidate_transcript)
    finally:
        os.unlink(audio_path)

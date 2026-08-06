import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from approach_1 import config
from approach_1.src.evaluate import evaluate
from approach_1.src.models import EvaluationInputs, EvaluationReportV2

app = FastAPI(title="Audio vs Transcript Evaluator")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluationReportV2)
def evaluate_endpoint(
    audio: UploadFile = File(...),
    gold_transcript: str = Form(...),
) -> EvaluationReportV2:
    stt_runner = config.get_stt_runner()
    judge = config.get_judge()
    embedder = config.get_embedder()

    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio.file.read())
        audio_path = tmp.name
    if audio_path is None:
        raise RuntimeError("Failed to create temporary audio file")

    try:
        candidate_transcript = stt_runner.transcribe(audio_path)
        return evaluate(
            gold_transcript,
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
    finally:
        os.unlink(audio_path)


@app.post("/evaluate-text", response_model=EvaluationReportV2)
def evaluate_text(
    gold_transcript: str = Form(...),
    candidate_transcript: str = Form(...),
) -> EvaluationReportV2:
    """Compare two texts directly (no audio/STT), for offline testing."""
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

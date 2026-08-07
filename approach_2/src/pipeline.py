"""End-to-end pipeline: align -> compare -> score -> sample -> ReviewReport.

`build_report` is the testable core (synthetic segments). `run_pipeline` loads
the stored per-engine transcripts for a dataset file and evaluates them, so the
review step does not re-run any STT engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from approach_2 import config
from approach_2.src.align import align
from approach_2.src.compare import compare
from approach_2.src.models import EngineSegment, ReviewReport, SpotCheck
from approach_2.src.review import sample_review_set
from approach_2.src.score import score


def load_segments(engine: str, stem: str) -> list[EngineSegment]:
    path = config.OUTPUT_DIRS[engine] / f"{stem}.segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EngineSegment.model_validate(d) for d in data]


def verdicts_path(stem: str) -> Path:
    """JSON sidecar storing reviewer verdicts/corrections per segment."""
    return config.DATASET_DIR / "review" / stem / "verdicts.json"


def load_verdicts(stem: str) -> dict[int, dict]:
    path = verdicts_path(stem)
    if not path.is_file():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def save_verdicts(stem: str, verdicts: dict[int, dict]) -> None:
    path = verdicts_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({str(k): v for k, v in verdicts.items()}, indent=2),
        encoding="utf-8",
    )


def build_report(
    segments_a: list[EngineSegment],
    segments_b: list[EngineSegment],
    audio: str,
    seed: int | None = None,
    fraction: float | None = None,
) -> ReviewReport:
    """Evaluate two segment streams into a complete ReviewReport."""
    aligned = align(segments_a, segments_b)
    for seg in aligned:
        compare(seg)
        score(seg)
    sample_ids = sample_review_set(aligned, seed=seed, fraction=fraction)
    return ReviewReport(
        audio=audio,
        engines=[config.ENGINE_A, config.ENGINE_B],
        segments=aligned,
        spot_check=SpotCheck(seed=config.SPOT_CHECK_SEED if seed is None else seed, sample_ids=sample_ids),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def run_pipeline(stem: str, seed: int | None = None) -> ReviewReport:
    """Evaluate a dataset file's stored transcripts (no re-transcription)."""
    segments_a = load_segments(config.ENGINE_A, stem)
    segments_b = load_segments(config.ENGINE_B, stem)
    return build_report(segments_a, segments_b, audio=stem, seed=seed)

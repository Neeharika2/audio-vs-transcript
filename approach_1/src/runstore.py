"""Shared persistence for reviewable evaluation runs.

Both the web API and the CLI write comparison results here, so anything run
with `python -m approach_1.main evaluate|evaluate-text` also shows up on the
`/review` page. Each run is a single JSON payload:

    datasets/review/<run_id>.json     stored run payload
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REVIEW_DIR = BASE / "datasets" / "review"


def save_run(report, run_id: str, gold: str, candidate: str) -> dict:
    """Persist an evaluation result as a reviewable run.

    `report` must be an `EvaluationReportV2` (or anything with `.model_dump` and
    `.inputs.stt_model`). Returns the stored payload. Audio is intentionally NOT
    copied — the stored audio already lives in `datasets/recordings/`.
    """
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "label": run_id,
        "gold": gold,
        "candidate": candidate,
        "stt_model": (report.inputs.stt_model if report.inputs else None),
        "report": report.model_dump(mode="json"),
    }
    (REVIEW_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_run(run_id: str) -> dict | None:
    """Return the stored payload for a run, or None."""
    path = REVIEW_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs() -> list[dict]:
    """Summaries of every stored run, newest first."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(REVIEW_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("report", {})
        runs.append({
            "id": payload["id"],
            "label": payload.get("label", payload["id"]),
            "created_at": payload.get("created_at"),
            "stt_model": report.get("inputs", {}).get("stt_model"),
            "overall_score": report.get("overall_score"),
            "status": report.get("status"),
        })
    runs.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return runs
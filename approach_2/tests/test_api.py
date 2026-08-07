"""FastAPI review endpoints: list files, fetch reports, post/clear verdicts."""

import json

from fastapi.testclient import TestClient

from approach_2 import config
from approach_2.api import app

client = TestClient(app)


def test_files_lists_audio_stems():
    stems = client.get("/files").json()
    assert isinstance(stems, list)
    assert all(isinstance(s, str) and s for s in stems)


def test_report_endpoint_returns_scored_segments():
    has_transcripts = (
        config.OUTPUT_DIRS[config.ENGINE_A] / "audio-1.segments.json"
    ).is_file() and (config.OUTPUT_DIRS[config.ENGINE_B] / "audio-1.segments.json").is_file()
    if not has_transcripts:
        import pytest

        pytest.skip("dataset transcripts not present")
    body = client.get("/report/audio-1").json()
    assert body["segments"]
    assert all(0 <= s["confidence"] <= 100 for s in body["segments"])
    assert all(s["agreement"] is not None for s in body["segments"])


def test_review_post_clear_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path)
    stem = "audio-1"

    def verdicts() -> dict:
        path = config.DATASET_DIR / "review" / stem / "verdicts.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    assert verdicts() == {}
    resp = client.post(f"/review/{stem}", json={"idx": 0, "verdict": "correct"})
    assert resp.status_code == 200
    assert verdicts() == {"0": {"verdict": "correct"}}
    assert resp.json()["segments"][0]["verdict"] == "correct"

    resp = client.post(f"/review/{stem}", json={"idx": 0, "verdict": ""})
    assert resp.status_code == 200
    # clearing a verdict must not leave an empty entry behind
    assert verdicts() == {}
    assert resp.json()["segments"][0]["verdict"] == "unreviewed"

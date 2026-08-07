# Audio vs Transcript — one-command entry point.
#
#   make            show this menu
#   make setup      one-time install (venv + dependencies)
#   make run        run the whole approach_2 pipeline, then open the web UI
#   make tests      run the full test suite
#   make eval1      run approach_1 on one gold-reference file
#
# The full pipeline "run" does, in order:
#   1. transcribing every audio file with 2 STT engines (Whisper + Deepgram)
#   2. building the agreement/diff reports
#   3. having the LLM judge judge the flagged segments
#   4. launching the interactive review UI at http://localhost:8000

PY := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn

.PHONY: help setup run transcribe review judge ui tests eval1 clean

help: ## show this menu
	@echo "Usage:  make [target]"
	@echo ""
	@echo "  make run       whole pipeline + web UI  (this is all most users need)"
	@echo "  make setup     one-time: create .venv + install dependencies"
	@echo "  make ui        open the review UI  (http://localhost:8000)"
	@echo "  make tests     run the test suite"
	@echo "  make eval1     approach_1 (needs a gold transcript file)"
	@echo "  Partial steps (used by 'run', callable on their own):"
	@echo "    make transcribe   transcribe audio with both STT engines"
	@echo "    make review       build reports from existing transcripts"
	@echo "    make judge        LLM-judge flagged segments"

setup:
	@command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg is required but not found."; exit 1; }
	@test -f .venv/bin/python || python -m venv .venv
	@$(PIP) install -r approach_2/requirements.txt
	@test -f .env || { echo "WARNING: no .env found — copy one with DEEPGRAM_API_KEY / GEMINI_API_KEY."; exit 1; }
	@echo "Setup complete. Run 'make run'."

transcribe:
	@echo "== [1/3] Transcribing all audio with Whisper + Deepgram =="
	@$(PY) -m approach_2.main transcribe

review:
	@echo "== [2/3] Building deterministic reports (diff + agreement + tiers) =="
	@$(PY) -m approach_2.main review

judge:
	@echo "== [3/3] LLM-judging flagged segments =="
	@$(PY) -m approach_2.main judge

ui:
	@echo "== Opening review UI at http://localhost:8000 =="
	@$(UVICORN) approach_2.api:app --port 8000

run: setup transcribe review judge
	@make ui

eval1:
	@test -n "$(AUDIO)" -a -n "$(GOLD)" || { echo "Usage:  make eval1 AUDIO=path/file.wav GOLD=path/gold.txt"; exit 1; }
	@echo "== approach_1: transcribe $(AUDIO) and grade it against $(GOLD) =="
	@$(PY) -m approach_1.main evaluate $(AUDIO) --gold $(GOLD)

tests:
	@$(PY) -m pytest approach_1/tests approach_2/tests -q

layer3:
	@$(PY) -m pytest -m layer3 -q

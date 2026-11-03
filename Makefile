# Convenience targets. All commands run from the repository root.
PY ?= python3
export PYTHONPATH := src

.PHONY: test lint format verify gold-smoke gold serve

test:            ## unit, contract and integration tests
	$(PY) -m unittest discover -s tests -v

lint:            ## style and static checks
	ruff check src scripts tests
	black --check src scripts tests

format:          ## auto-format
	black src scripts tests
	ruff check --fix src scripts tests

verify:          ## frozen corpus snapshot: hashes + byte-identical re-extraction
	$(PY) scripts/verify_freeze.py

gold-smoke:      ## Part 3 pipeline on 5 items with the local model
	$(PY) scripts/run_gold_llm.py --limit 5

gold:            ## Part 3 full run (local Ollama model, resumable)
	$(PY) scripts/run_gold_llm.py

serve:           ## local FastAPI demo on http://127.0.0.1:8765
	$(PY) scripts/serve.py

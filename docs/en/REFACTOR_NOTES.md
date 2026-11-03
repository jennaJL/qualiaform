# v3 refactor: what changed and how behaviour was verified

## Why

The v2 code worked, but it was written in a compressed style: several
statements per line, single-letter names, no docstrings. Reviewers could not
read it. The refactor makes the code readable without changing any result.

## What changed

- Every module was rewritten with descriptive names, docstrings that explain
  the design decisions (masking, dev-only selection, trie-constrained
  decoding, out-of-fold demos), type hints, and long functions split into
  steps: `_classify_form`, `_train_head`, `_lexical_baseline`,
  `_run_identity`, and so on.
- `black` and `ruff` are configured in `pyproject.toml` and enforced in CI;
  `make lint` and `make format` run them locally.
- The unused `fast_decode.py` development experiment was removed from the
  package. It remains in the archive.
- Documentation was rewritten in English in `docs/en/`.
- New: `src/qualiaform_part2/gold/` (Part 3), `scripts/run_gold_llm.py`,
  `tests/test_gold.py` (15 tests), and a `Makefile`.

## How "no behaviour change" was checked

| Component | Check | Result |
|---|---|---|
| `extract.py` | Re-extract all 9,153 candidates; compare with the frozen snapshot | `candidates.jsonl`, `candidates.csv`, `statistics.json` byte-identical |
| `evaluate.py` | Re-score all 5 saved LLM test runs | `metrics.json` and `errors.csv` byte-identical |
| `experiment.py` | Re-run the 4 mock conditions and `prepare` mode | `requests.jsonl` and `predictions.jsonl` byte-identical |
| `portfolio.py` | Train old and new code on the same machine from the frozen features | all weights, predictions, metrics, bootstrap and slices byte-identical (manifest differs only in code hash and run time) |
| `local_cpu.py` | Rebuild the requests of the saved test runs | identical to the stored `requests.jsonl` |
| everything | the original 53 tests | pass |

Result manifests store SHA-256 hashes of the code that produced them. The
pre-refactor code is therefore kept unchanged in `archive/v2_original_code/`,
and the provenance checks (`scripts/verify_freeze.py`,
`scripts/report_local_llm.py`) verify against it.

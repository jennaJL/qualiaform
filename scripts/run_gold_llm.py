"""One command for the gold-set LLM experiment (Part 3).

Steps: prepare inputs (if needed) → check the local model → run every
condition (resumable) → analyse → write a report.

Requirements: Python 3.10+, numpy, scikit-learn, and Ollama running locally
with the model pulled (``ollama pull qwen2.5:7b-instruct``). Free, no API key.

Examples::

    python scripts/run_gold_llm.py                       # full run, default model
    python scripts/run_gold_llm.py --limit 5             # quick smoke test
    python scripts/run_gold_llm.py --backend mock        # pipeline test, no model
    python scripts/run_gold_llm.py --model qwen3:8b      # another local model

Interrupting is safe: rerun the same command to continue.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qualiaform_part2.extract import dump, read_rows  # noqa: E402
from qualiaform_part2.gold.analyze import analyse, load_runs, write_item_table  # noqa: E402
from qualiaform_part2.gold.llm import (  # noqa: E402
    CONDITIONS,
    DEFAULT_MODEL,
    MockClient,
    OllamaClient,
    run_condition,
)
from qualiaform_part2.gold.report import write_report  # noqa: E402

ITEMS = ROOT / "data/private/gold_v1/items.jsonl"


def prepare_inputs() -> None:
    print("Preparing gold-set inputs …")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "qualiaform_part2.gold.prepare"], check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--limit", type=int, help="only the first N items (smoke test, no analysis)")
    parser.add_argument("--out", help="run folder (default: data/private/gold_runs/<model>)")
    args = parser.parse_args()

    if not ITEMS.exists():
        prepare_inputs()
    all_items = read_rows(ITEMS)
    items = all_items[: args.limit] if args.limit else all_items

    client = MockClient() if args.backend == "mock" else OllamaClient(args.model)
    try:
        model_info = client.describe()  # fails early, with instructions, if the model is missing
    except RuntimeError as exc:
        raise SystemExit(f"\n{exc}\n") from None
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_info["model"])
    if args.limit:
        tag += f"_limit{args.limit}"
    out = pathlib.Path(args.out) if args.out else ROOT / "data/private/gold_runs" / tag
    print(f"Model:  {json.dumps(model_info)}\nItems:  {len(items)}\nOutput: {out}\n")

    start = time.monotonic()
    for condition in args.conditions:
        print(f"Running {condition} …", flush=True)
        run_condition(condition, items, client, out, progress=lambda m: print(m, flush=True))
    print(f"\nAll conditions done in {(time.monotonic() - start) / 60:.1f} min.")
    if args.limit:
        print("Smoke test complete. Rerun without --limit for the full experiment.")
        return

    summary, systems = analyse(items, load_runs(out))
    summary["model"] = model_info
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    dump(analysis / "summary.json", summary)
    write_report(summary, analysis / "report.md")
    write_item_table(analysis / "items_with_predictions_PRIVATE.csv", items, load_runs(out), systems)

    if args.backend != "mock":
        public = ROOT / "reports/gold_llm" / tag
        public.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "report.md"):  # labels and scores only, no source text
            shutil.copy(analysis / name, public / name)
        print(f"\nReport (shareable): {public / 'report.md'}")
    print(
        f"Item-level table (private, contains source text): {analysis / 'items_with_predictions_PRIVATE.csv'}"
    )


if __name__ == "__main__":
    main()

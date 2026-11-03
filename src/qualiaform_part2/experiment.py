"""Build reproducible LLM requests and (optionally) run them.

Three modes:

* ``prepare``  – write the exact requests only (no model is called).
* ``mock``     – deterministic, deliberately meaningless predictions used to
  smoke-test the data flow. Mock scores say nothing about any model.
* ``external`` – pipe each request (JSON on stdin) to a user-supplied command
  and read one JSON object from its stdout.

Only the fields in :data:`SAFE_FIELDS` are ever sent to a model, so the
surface form (the answer) cannot leak into the prompt.

Usage::

    python -m qualiaform_part2.experiment --config configs/direct_zero.json \
        --mode mock --out results/my_mock_direct_zero
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any

from .extract import LABELS, digest, dump, read_rows, write_rows

SAFE_FIELDS = ("masked_context", "noun_lemma", "verb_lemma", "relation")
TYPICALITY_VALUES = ("typical", "atypical", "uncertain")
QUALIA_VALUES = ("TELIC", "AGENTIVE", "HAVE", "NONE")
DEMO_PROVENANCE = ("part1_gold", "synthetic_smoke")


def model_input(row: dict[str, Any]) -> dict[str, Any]:
    """Return only the leakage-safe input fields of a candidate row."""
    return {key: row[key] for key in SAFE_FIELDS}


def validate_response(obj: Any, condition: str) -> dict[str, Any]:
    """Check a parsed model response against the output schema.

    ``direct`` responses need ``form_prediction``; ``structured`` responses
    additionally need ``typicality``, ``qualia`` and a string ``evidence``.
    """
    if not isinstance(obj, dict) or obj.get("form_prediction") not in LABELS:
        raise ValueError("invalid form_prediction")
    if condition == "structured":
        if obj.get("typicality") not in TYPICALITY_VALUES:
            raise ValueError("invalid typicality")
        if obj.get("qualia") not in QUALIA_VALUES:
            raise ValueError("invalid qualia")
        if not isinstance(obj.get("evidence"), str):
            raise ValueError("missing evidence")
    return obj


def demonstrations(
    path: str | pathlib.Path,
    targets: list[dict[str, Any]],
    condition: str,
    allow_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Load few-shot demonstrations and refuse any that overlap ``targets``.

    Each demonstration must declare its provenance, use only safe input
    fields, and must not share an ID, sentence or masked context with any
    evaluation target.
    """
    demos = read_rows(path)
    if not demos:
        raise ValueError("few-shot requires nonempty demonstrations")
    target_contexts = {r["masked_context"] for r in targets}
    target_hashes = {r["sentence_hash"] for r in targets}
    target_ids = {r["candidate_id"] for r in targets}

    for demo in demos:
        provenance = demo.get("provenance")
        if provenance not in DEMO_PROVENANCE:
            raise ValueError("demonstration provenance required")
        if provenance == "synthetic_smoke" and not allow_synthetic:
            raise ValueError("synthetic examples allowed only in mock smoke runs")
        if (
            demo.get("candidate_id") in target_ids
            or demo.get("sentence_hash") in target_hashes
            or demo["input"]["masked_context"] in target_contexts
        ):
            raise ValueError("demonstration overlaps evaluation target")
        if set(demo["input"]) != set(SAFE_FIELDS):
            raise ValueError("unsafe demonstration fields")
        if provenance == "part1_gold" and not demo.get("sentence_hash"):
            raise ValueError("gold demo needs original sentence hash for overlap checks")
        validate_response(demo["output"], condition)

    def output_for(demo: dict[str, Any]) -> dict[str, Any]:
        if condition == "structured":
            return demo["output"]
        return {"form_prediction": demo["output"]["form_prediction"]}

    return [{"input": d["input"], "output": output_for(d)} for d in demos]


def select_rows(all_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Rows of ``split``, minus sentences that also occur in an earlier split.

    Priority is train > dev > test, so a test sentence that also occurs in
    train or dev is dropped. Exact (sentence, noun) duplicates are removed.
    """
    train_hashes = {r["sentence_hash"] for r in all_rows if r["source_split"] == "train"}
    dev_hashes = {r["sentence_hash"] for r in all_rows if r["source_split"] == "dev"}
    excluded = {"test": train_hashes | dev_hashes, "dev": train_hashes}.get(split, set())

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in all_rows:
        if row["source_split"] != split or row["sentence_hash"] in excluded:
            continue
        key = (row["sentence_hash"], row["noun_token_id"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def mock_response(row: dict[str, Any], condition: str) -> dict[str, Any]:
    """A deterministic pseudo-random answer that never looks at the label."""
    seed = digest(json.dumps(model_input(row), sort_keys=True).encode())
    result: dict[str, Any] = {"form_prediction": LABELS[int(seed[:8], 16) % 2]}
    if condition == "structured":
        result.update(typicality="uncertain", qualia="NONE", evidence="MOCK: no semantic inference")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run LLM requests.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="data/processed/ud_candidates_v1/candidates.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["prepare", "mock", "external"], default="prepare")
    parser.add_argument("--demo-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--adapter", nargs="+", help="external command, e.g. python my_model.py")
    args = parser.parse_args()

    cfg = json.loads(pathlib.Path(args.config).read_text())
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("Refusing to overwrite experiment")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.mode == "external" and (not args.adapter or not args.model_id):
        raise ValueError("external requires --adapter and --model-id")

    all_rows = read_rows(args.dataset)
    rows = select_rows(all_rows, cfg["split"])
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        raise ValueError("empty evaluation set")

    prompt = pathlib.Path(cfg["prompt"]).read_text(encoding="utf8")
    demos: list[dict[str, Any]] = []
    if cfg["shots"] == "few":
        if not args.demo_file:
            raise ValueError("few-shot requires --demo-file; Part 1 gold unavailable")
        demos = demonstrations(
            args.demo_file, all_rows, cfg["condition"], allow_synthetic=args.mode == "mock"
        )

    out.mkdir(parents=True)
    requests: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for row in rows:
        request = {
            "system": prompt,
            "demonstrations": demos,
            "input": model_input(row),
            "generation": cfg["generation"],
        }
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
        request_hash = digest(encoded.encode())
        requests.append(
            {"candidate_id": row["candidate_id"], "request_sha256": request_hash, "request": request}
        )
        if args.mode == "prepare":
            continue

        raw, error, result = "", None, None
        try:
            if args.mode == "mock":
                raw = json.dumps(mock_response(row, cfg["condition"]), ensure_ascii=False)
            else:
                process = subprocess.run(
                    args.adapter,
                    input=encoded,
                    text=True,
                    capture_output=True,
                    timeout=cfg["timeout_seconds"],
                    check=True,
                )
                raw = process.stdout
            result = validate_response(json.loads(raw), cfg["condition"])
        except (ValueError, subprocess.SubprocessError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = None
        predictions.append(
            {
                "candidate_id": row["candidate_id"],
                "prediction": result,
                "raw_response": raw,
                "error": error,
                "request_sha256": request_hash,
                "execution_mode": args.mode,
            }
        )

    write_rows(out / "requests.jsonl", requests)
    if args.mode != "prepare":
        write_rows(out / "predictions.jsonl", predictions)
    dump(
        out / "manifest.json",
        {
            "execution_mode": args.mode,
            "scientific_model_experiment_run": args.mode == "external",
            "model_id": args.model_id if args.mode == "external" else "NO_MODEL_" + args.mode.upper(),
            "config": cfg,
            "config_sha256": digest(pathlib.Path(args.config).read_bytes()),
            "prompt_sha256": digest(prompt.encode()),
            "dataset_sha256": digest(pathlib.Path(args.dataset).read_bytes()),
            "demonstration_sha256": (
                digest(pathlib.Path(args.demo_file).read_bytes()) if args.demo_file else None
            ),
            "count": len(rows),
            "adapter_command": args.adapter,
            "code_sha256": digest(pathlib.Path(__file__).read_bytes()),
            "files": {f.name: digest(f.read_bytes()) for f in out.iterdir()},
        },
    )
    print(f"{args.mode}: {len(rows)} requests; {out}")


if __name__ == "__main__":
    main()

"""Classification metrics and evaluation of saved LLM prediction runs.

Design rules:

* Predictions are joined to references strictly by ``candidate_id``.
* Invalid or unparsable model outputs are kept and counted as errors
  (column ``INVALID`` in the confusion matrix) – never silently dropped.
* No semantic gold is invented: errors are exported for human review with
  empty reviewer fields.

Usage::

    python -m qualiaform_part2.evaluate --run results/local_v2_test_direct_zero
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from .extract import LABELS, digest, dump, read_rows

INVALID = "INVALID"
ERROR_FIELDS = [
    "candidate_id",
    "sentence",
    "reference_observed_form",
    "prediction",
    "predicted_typicality",
    "predicted_qualia",
    "flags",
    "error_type",
    "reviewer",
    "review_notes",
    "indefinite_gold",
]


def metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro-F1 and per-class scores.

    ``pairs`` is a sequence of ``(reference, prediction)`` labels. Any
    prediction outside :data:`LABELS` is counted as ``INVALID``.

    Undefined precision/F1 (no predictions for a class) is reported as 0.0.
    Balanced accuracy is ``None`` if either reference class is absent.
    """
    matrix = {ref: {pred: 0 for pred in (*LABELS, INVALID)} for ref in LABELS}
    for reference, prediction in pairs:
        if reference not in LABELS:
            raise ValueError("unknown reference label")
        matrix[reference][prediction if prediction in LABELS else INVALID] += 1

    n = len(pairs)
    per_class = {}
    for label in LABELS:
        true_positives = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[ref][label] for ref in LABELS)
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / support if support else None
        if recall is not None and precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "n": n,
        "accuracy": sum(matrix[label][label] for label in LABELS) / n if n else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if len(recalls) == 2 else None,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2 if n else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "invalid": sum(matrix[label][INVALID] for label in LABELS),
        "metric_policy": "zero precision/F1 for undefined denominator; "
        "balanced accuracy null if either reference class absent",
    }


def evaluate(rows: list[dict], predictions: list[dict]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score ``predictions`` against candidate ``rows``.

    Returns ``(results, errors)`` where ``results`` holds overall metrics and
    metrics per slice (split, relation, noun, predicted semantics), and
    ``errors`` lists every misclassified or invalid item for review.
    """
    by_id = {r["candidate_id"]: r for r in rows}
    ids = [p["candidate_id"] for p in predictions]
    if len(by_id) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("duplicate ID")
    if set(ids) - set(by_id):
        raise ValueError("unknown prediction ID")

    pairs: list[tuple[str, str]] = []
    errors: list[dict[str, Any]] = []
    slices: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for prediction in predictions:
        row = by_id[prediction["candidate_id"]]
        output = prediction.get("prediction") or {}
        predicted = output.get("form_prediction", INVALID)
        reference = row["observed_form"]
        pair = (reference, predicted)
        pairs.append(pair)

        for field in ("source_split", "relation", "noun_lemma"):
            slices[f"{field}:{row[field]}"].append(pair)
        for field in ("typicality", "qualia"):
            slices[f"PREDICTED_{field}:{output.get(field, 'unavailable')}"].append(pair)

        if reference != predicted:
            errors.append(
                {
                    "candidate_id": row["candidate_id"],
                    "sentence": row["sentence"],
                    "reference_observed_form": reference,
                    "prediction": predicted,
                    "predicted_typicality": output.get("typicality"),
                    "predicted_qualia": output.get("qualia"),
                    "flags": "|".join(row["flags"]),
                    "error_type": "invalid_response" if predicted not in LABELS else "unreviewed",
                    "reviewer": "",
                    "review_notes": "",
                    "indefinite_gold": "unavailable",
                }
            )

    results = {
        "overall": metrics(pairs),
        "slices": {name: metrics(group) for name, group in sorted(slices.items())},
    }
    return results, errors


def _check_run_integrity(run: pathlib.Path, dataset: pathlib.Path, manifest: dict) -> list[dict]:
    """Verify that a run directory is complete and matches the dataset."""
    if digest(dataset.read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    predictions = read_rows(run / "predictions.jsonl")
    requests = read_rows(run / "requests.jsonl")
    if {x["candidate_id"] for x in predictions} != {x["candidate_id"] for x in requests}:
        raise ValueError("missing/extra predictions")
    if len(predictions) != manifest["count"] or len(requests) != manifest["count"]:
        raise ValueError("run count mismatch")
    request_hash = {x["candidate_id"]: x["request_sha256"] for x in requests}
    if any(x["request_sha256"] != request_hash[x["candidate_id"]] for x in predictions):
        raise ValueError("request mismatch")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved prediction run.")
    parser.add_argument("--dataset", default="data/processed/ud_candidates_v1/candidates.jsonl")
    parser.add_argument("--run", required=True, help="folder with manifest/requests/predictions")
    args = parser.parse_args()

    run = pathlib.Path(args.run)
    dataset = pathlib.Path(args.dataset)
    manifest = json.loads((run / "manifest.json").read_text())
    predictions = _check_run_integrity(run, dataset, manifest)

    results, errors = evaluate(read_rows(dataset), predictions)
    results.update(
        execution_mode=manifest["execution_mode"],
        interpretation=(
            "SMOKE ONLY; not LLM performance"
            if manifest["execution_mode"] == "mock"
            else "Agreement with extracted observed form; not indefinite-gold accuracy"
        ),
    )
    dump(run / "metrics.json", results)
    with (run / "errors.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps(results["overall"], indent=2))


if __name__ == "__main__":
    main()

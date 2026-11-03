"""Analyse gold-set LLM runs: annotation agreement and form-prediction comparison.

Three questions:

1. **Annotation** – how well does the LLM reproduce the expert qualia labels?
   (accuracy, Cohen's kappa, confusion matrix, per-label precision/recall)
2. **Form prediction** – on the same 155 items and the same Part 1 folds,
   compare: majority; LLM direct; LLM structured; a theory rule; and
   logistic regression on context features with no / gold / LLM qualia.
3. **Licensing** – is P(BN | qualia) higher when the verb matches a qualia
   role than with NONE (the asymmetry the theory predicts), using gold and
   LLM labels?

All logistic-regression scores are out-of-fold (Part 1 grouped folds, seed 42,
class-balanced, C=1, threshold 0.5 – the Part 1 protocol). Uncertainty is
given as item-level bootstrap 95% intervals, and paired bootstrap intervals
for differences between systems evaluated on the same items.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
from typing import Any

import numpy as np

from ..evaluate import metrics
from ..extract import dump, read_rows
from .codebook import QUALIA, TYPICAL_QUALIA
from .llm import CONDITIONS, split_condition

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONTEXT_FEATURES = ["neg_dis", "modifier", "aspect"]  # Part 1 "context" feature set
LICENSING_QUALIA = ("TELIC", "AGENTIVE", "HAVE")
BOOTSTRAP_SAMPLES = 2000


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_runs(run_dir: pathlib.Path) -> dict[str, dict[str, dict | None]]:
    """``{condition: {item_id: parsed prediction or None}}`` for completed runs."""
    runs = {}
    for condition in CONDITIONS:
        manifest = run_dir / condition / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["completed"]:
            rows = read_rows(run_dir / condition / "predictions.jsonl")
            runs[condition] = {r["item_id"]: r["prediction"] for r in rows}
    return runs


# --------------------------------------------------------------------------
# Annotation agreement
# --------------------------------------------------------------------------


def cohen_kappa(reference: list[str], predicted: list[str], labels: list[str]) -> float:
    """Cohen's kappa for two raters over ``labels`` (invalid answers count as their own label)."""
    categories = list(labels) + sorted(set(predicted) - set(labels))
    index = {label: i for i, label in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)))
    for ref, pred in zip(reference, predicted):
        table[index[ref], index[pred]] += 1
    n = table.sum()
    observed = np.trace(table) / n
    expected = (table.sum(1) @ table.sum(0)) / n**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def agreement(items: list[dict], predictions: dict[str, dict | None]) -> dict[str, Any]:
    """Qualia agreement with gold, plus agreement on the derived typicality."""
    gold = [i["qualia_gold"] for i in items]
    llm = [(predictions[i["item_id"]] or {}).get("qualia", "INVALID") for i in items]
    confusion = {g: {p: 0 for p in (*QUALIA, "INVALID")} for g in QUALIA}
    for g, p in zip(gold, llm):
        confusion[g][p if p in QUALIA else "INVALID"] += 1
    per_label = {}
    for label in QUALIA:
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in QUALIA)
        support = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    gold_typical = ["1" if q in TYPICAL_QUALIA else "0" for q in gold]
    llm_typical = ["1" if q in TYPICAL_QUALIA else "0" for q in llm]
    return {
        "n": len(items),
        "accuracy": sum(g == p for g, p in zip(gold, llm)) / len(items),
        "kappa": cohen_kappa(gold, llm, list(QUALIA)),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(QUALIA),
        "per_label": per_label,
        "confusion": confusion,
        "invalid": sum(p not in QUALIA for p in llm),
        "typicality_accuracy": sum(g == p for g, p in zip(gold_typical, llm_typical)) / len(items),
        "typicality_kappa": cohen_kappa(gold_typical, llm_typical, ["0", "1"]),
        "predicted_distribution": dict(collections.Counter(llm)),
    }


# --------------------------------------------------------------------------
# Form prediction
# --------------------------------------------------------------------------


def design_matrix(items: list[dict], qualia: list[str] | None) -> np.ndarray:
    """Context features, plus one-hot qualia with NONE as the reference level."""
    columns = [[item[f] for f in CONTEXT_FEATURES] for item in items]
    if qualia is not None:
        for row, label in zip(columns, qualia):
            row.extend(int(label == q) for q in ("HAVE", "TELIC", "AGENTIVE"))
    return np.array(columns, dtype=float)


def oof_logistic(items: list[dict], qualia: list[str] | None) -> list[str]:
    """Out-of-fold class-balanced logistic regression (Part 1 protocol)."""
    from sklearn.linear_model import LogisticRegression

    x = design_matrix(items, qualia)
    y = np.array([int(i["form_target"] == "BN") for i in items])
    folds = np.array([i["fold"] for i in items])
    predicted = np.empty(len(items), dtype=object)
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        model = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=3000)
        model.fit(x[train], y[train])
        bn_probability = model.predict_proba(x[test])[:, list(model.classes_).index(1)]
        predicted[test] = np.where(bn_probability >= 0.5, "BN", "YI_CL_N")
    return list(predicted)


def licensing_rule(qualia: list[str]) -> list[str]:
    """Theory rule: predict BN iff the verb matches TELIC, AGENTIVE or HAVE."""
    return ["BN" if q in LICENSING_QUALIA else "YI_CL_N" for q in qualia]


def llm_qualia(items: list[dict], predictions: dict[str, dict | None]) -> list[str]:
    """LLM qualia labels; an invalid answer is treated as NONE (the default class)."""
    return [(predictions[i["item_id"]] or {}).get("qualia") or "NONE" for i in items]


def form_systems(items: list[dict], runs: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Every system's predicted forms, with a short description."""
    gold_qualia = [i["qualia_gold"] for i in items]
    systems: dict[str, dict[str, Any]] = {
        "majority (always YI_CL_N)": {"kind": "baseline", "pred": ["YI_CL_N"] * len(items)},
        "logreg: context only": {"kind": "classifier", "pred": oof_logistic(items, None)},
        "logreg: context + gold qualia (oracle)": {
            "kind": "oracle",
            "pred": oof_logistic(items, gold_qualia),
        },
        "rule: gold qualia licenses BN (oracle)": {"kind": "oracle", "pred": licensing_rule(gold_qualia)},
    }
    for condition, predictions in runs.items():
        task, shots = split_condition(condition)
        if task in ("direct", "direct_reason", "structured"):
            systems[f"LLM {condition}"] = {
                "kind": "llm",
                "pred": [(predictions[i["item_id"]] or {}).get("form", "INVALID") for i in items],
            }
        if task in ("annotate", "structured"):
            qualia = llm_qualia(items, predictions)
            systems[f"logreg: context + LLM qualia ({condition})"] = {
                "kind": "pipeline",
                "pred": oof_logistic(items, qualia),
            }
            systems[f"rule: LLM qualia licenses BN ({condition})"] = {
                "kind": "pipeline",
                "pred": licensing_rule(qualia),
            }
    return systems


def _balanced_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    recalls = [np.mean(pred[gold == label] == label) for label in ("BN", "YI_CL_N")]
    return float(np.mean(recalls))


def bootstrap_intervals(gold: list[str], systems: dict[str, dict], pairs: list[tuple[str, str]]) -> dict:
    """Item-level bootstrap: 95% CI of balanced accuracy, and of paired differences."""
    rng = np.random.default_rng(1729)
    gold_array = np.array(gold)
    preds = {name: np.array(s["pred"]) for name, s in systems.items()}
    samples = []
    while len(samples) < BOOTSTRAP_SAMPLES:
        idx = rng.integers(0, len(gold), len(gold))
        if len(set(gold_array[idx])) == 2:
            samples.append(idx)
    scores = {
        name: np.array([_balanced_accuracy(gold_array[i], p[i]) for i in samples])
        for name, p in preds.items()
    }
    intervals = {
        name: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for name, v in scores.items()
    }
    differences = {}
    for a, b in pairs:
        if a in scores and b in scores:
            delta = scores[a] - scores[b]
            differences[f"{a}  minus  {b}"] = {
                "observed": _balanced_accuracy(gold_array, preds[a])
                - _balanced_accuracy(gold_array, preds[b]),
                "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
                "share_of_resamples_above_zero": float(np.mean(delta > 0)),
            }
    return {"balanced_accuracy_ci95": intervals, "paired_differences": differences}


KEY_COMPARISONS = [
    ("LLM structured_few", "LLM direct_few"),
    ("LLM structured_zero", "LLM direct_zero"),
    ("LLM structured_zero", "LLM direct_reason_zero"),
    ("logreg: context + LLM qualia (annotate_few)", "logreg: context only"),
    ("logreg: context + gold qualia (oracle)", "logreg: context only"),
    ("logreg: context + gold qualia (oracle)", "logreg: context + LLM qualia (annotate_few)"),
    ("logreg: context + LLM qualia (annotate_few)", "LLM direct_few"),
]


def licensing_table(items: list[dict], qualia: list[str]) -> dict[str, dict[str, Any]]:
    """BN rate per qualia label: the theory predicts NONE has the lowest rate."""
    table = {}
    for label in QUALIA:
        forms = [i["form_target"] for i, q in zip(items, qualia) if q == label]
        table[label] = {
            "n": len(forms),
            "bn": forms.count("BN"),
            "bn_rate": forms.count("BN") / len(forms) if forms else None,
        }
    match = [i["form_target"] for i, q in zip(items, qualia) if q in LICENSING_QUALIA]
    none = [i["form_target"] for i, q in zip(items, qualia) if q not in LICENSING_QUALIA]
    table["any match (TELIC/AGENTIVE/HAVE)"] = {
        "n": len(match),
        "bn": match.count("BN"),
        "bn_rate": match.count("BN") / len(match) if match else None,
    }
    table["share of BN with a match"] = {
        "n": sum(i["form_target"] == "BN" for i in items),
        "bn_rate": None,
        "value": (
            (match.count("BN") / (match.count("BN") + none.count("BN")))
            if (match + none).count("BN")
            else None
        ),
    }
    return table


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def analyse(items: list[dict], runs: dict[str, dict]) -> dict[str, Any]:
    gold_forms = [i["form_target"] for i in items]
    systems = form_systems(items, runs)
    form_results = {}
    for name, system in systems.items():
        m = metrics(list(zip(gold_forms, system["pred"])))
        form_results[name] = {
            "kind": system["kind"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "bn_recall": m["per_class"]["BN"]["recall"],
            "yi_recall": m["per_class"]["YI_CL_N"]["recall"],
            "bn_precision": m["per_class"]["BN"]["precision"],
            "invalid": m["invalid"],
            "confusion": m["confusion_matrix"],
        }
    licensing = {"gold": licensing_table(items, [i["qualia_gold"] for i in items])}
    annotation = {}
    for condition, predictions in runs.items():
        task, _ = split_condition(condition)
        if task in ("annotate", "structured"):
            annotation[condition] = agreement(items, predictions)
            licensing[condition] = licensing_table(items, llm_qualia(items, predictions))
    return {
        "n_items": len(items),
        "forms": dict(collections.Counter(gold_forms)),
        "annotation_agreement": annotation,
        "form_prediction": form_results,
        "bootstrap": bootstrap_intervals(gold_forms, systems, KEY_COMPARISONS),
        "licensing": licensing,
        "protocol": "Part 1 grouped folds (seed 42); out-of-fold logistic regression (balanced, C=1, "
        "threshold 0.5); LLM few-shot demos drawn only from the other four folds",
    }, systems


def write_item_table(path: pathlib.Path, items: list[dict], runs: dict[str, dict], systems: dict) -> None:
    """Private, item-level table for manual error analysis (contains source text)."""
    columns = ["item_id", "fold", "verb", "noun", "masked_context", "form_target", "qualia_gold"]
    llm_columns = []
    for condition in runs:
        llm_columns += [f"{condition}:qualia", f"{condition}:form", f"{condition}:reasoning"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns + llm_columns + [f"pred:{name}" for name in systems])
        for k, item in enumerate(items):
            row = [item[c] for c in columns]
            for predictions in runs.values():
                p = predictions[item["item_id"]] or {}
                row += [p.get("qualia", ""), p.get("form", ""), p.get("reasoning", "")]
            row += [s["pred"][k] for s in systems.values()]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse gold-set LLM runs.")
    parser.add_argument("--items", default=str(ROOT / "data/private/gold_v1/items.jsonl"))
    parser.add_argument("--runs", required=True, help="folder containing one sub-folder per condition")
    parser.add_argument("--out", help="summary folder (default: <runs>/analysis)")
    args = parser.parse_args()

    items = read_rows(args.items)
    run_dir = pathlib.Path(args.runs)
    runs = load_runs(run_dir)
    if not runs:
        raise SystemExit(f"No completed runs in {run_dir}")
    summary, systems = analyse(items, runs)
    model = json.loads((run_dir / next(iter(runs)) / "identity.json").read_text())["model"]
    summary["model"] = model

    out = pathlib.Path(args.out) if args.out else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "summary.json", summary)
    write_item_table(out / "items_with_predictions_PRIVATE.csv", items, runs, systems)
    from .report import write_report

    write_report(summary, out / "report.md")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()

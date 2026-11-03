"""Render the gold-set analysis summary as a Markdown report (no source text)."""

from __future__ import annotations

import pathlib
from typing import Any

from .codebook import QUALIA


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def _form_table(summary: dict[str, Any]) -> list[str]:
    ci = summary["bootstrap"]["balanced_accuracy_ci95"]
    lines = [
        "| System | Type | Balanced acc. | 95% CI | Macro-F1 | BN recall | YI recall | Invalid |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    rows = sorted(summary["form_prediction"].items(), key=lambda kv: -(kv[1]["balanced_accuracy"] or 0))
    for name, r in rows:
        low, high = ci[name]
        lines.append(
            f"| {name} | {r['kind']} | {_pct(r['balanced_accuracy'])} | [{low:.3f}, {high:.3f}] | "
            f"{_pct(r['macro_f1'])} | {_pct(r['bn_recall'])} | {_pct(r['yi_recall'])} | {r['invalid']} |"
        )
    return lines


def _agreement_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Condition | Accuracy | Cohen's κ | Macro-F1 | Typicality κ | "
        + " | ".join(f"{q} F1" for q in QUALIA)
        + " | Invalid |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(QUALIA) + "---:|",
    ]
    for condition, a in summary["annotation_agreement"].items():
        per_label = " | ".join(_pct(a["per_label"][q]["f1"]) for q in QUALIA)
        lines.append(
            f"| {condition} | {_pct(a['accuracy'])} | {_pct(a['kappa'])} | {_pct(a['macro_f1'])} | "
            f"{_pct(a['typicality_kappa'])} | {per_label} | {a['invalid']} |"
        )
    return lines


def _confusion(a: dict[str, Any]) -> list[str]:
    header = "| gold \\ LLM | " + " | ".join(QUALIA) + " | INVALID |"
    lines = [header, "|---|" + "---:|" * (len(QUALIA) + 1)]
    for gold in QUALIA:
        cells = " | ".join(str(a["confusion"][gold][p]) for p in (*QUALIA, "INVALID"))
        lines.append(f"| **{gold}** | {cells} |")
    return lines


def _licensing(summary: dict[str, Any]) -> list[str]:
    sources = list(summary["licensing"])
    lines = ["| Qualia | " + " | ".join(f"{s}: BN / n (rate)" for s in sources) + " |"]
    lines.append("|---|" + "---|" * len(sources))
    for label in (*QUALIA, "any match (TELIC/AGENTIVE/HAVE)"):
        cells = []
        for source in sources:
            row = summary["licensing"][source][label]
            cells.append(f"{row['bn']} / {row['n']} ({_pct(row['bn_rate'])})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def write_report(summary: dict[str, Any], path: pathlib.Path) -> None:
    model = summary.get("model", {})
    best_annotation = max(
        summary["annotation_agreement"].items(), key=lambda kv: kv[1]["kappa"], default=(None, None)
    )
    lines = [
        "# Gold-set LLM evaluation",
        "",
        f"Model: **{model.get('model')}** ({model.get('backend')}, "
        f"{model.get('parameter_size') or 'n/a'}, {model.get('quantization') or 'n/a'}). "
        f"Items: {summary['n_items']} ({summary['forms'].get('BN', 0)} BN / "
        f"{summary['forms'].get('YI_CL_N', 0)} YI_CL_N).",
        "",
        f"Protocol: {summary['protocol']}.",
        "",
        "## 1. Can the LLM reproduce the expert qualia annotation?",
        "",
        *_agreement_table(summary),
        "",
    ]
    if best_annotation[0]:
        lines += [
            f"Confusion matrix for **{best_annotation[0]}** (highest κ):",
            "",
            *_confusion(best_annotation[1]),
            "",
        ]
    lines += [
        "## 2. Form prediction: direct vs. linguistically informed",
        "",
        "Balanced accuracy is the primary metric (BN is the minority class). "
        "*oracle* = uses the expert's gold qualia labels; *pipeline* = uses LLM-predicted qualia.",
        "",
        *_form_table(summary),
        "",
        "### Paired differences (bootstrap over items)",
        "",
        "| Comparison | Δ balanced acc. | 95% CI | Resamples with Δ > 0 |",
        "|---|---:|---|---:|",
    ]
    for name, d in summary["bootstrap"]["paired_differences"].items():
        lines.append(
            f"| {name} | {d['observed']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
            f"{d['share_of_resamples_above_zero']:.0%} |"
        )
    lines += [
        "",
        "## 3. Licensing asymmetry",
        "",
        "The theory predicts that BN needs a qualia/possessive match: the BN rate should be "
        "clearly lower for NONE than for TELIC, AGENTIVE or HAVE.",
        "",
        *_licensing(summary),
        "",
        "## Caveats",
        "",
        "- 155 items from one translated novel; 31 BN. Intervals are wide, so small differences "
        "are not evidence of a real effect.",
        "- The gold qualia labels come from a single annotator; there is no inter-annotator "
        "agreement yet, so the LLM is compared with one expert, not with a consensus.",
        "- Source sentences are copyrighted; this report contains labels and scores only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf8")

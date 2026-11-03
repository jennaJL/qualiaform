"""Turn the Part 1 gold annotation into leakage-safe LLM inputs.

For each gold item this produces:

* ``verb`` / ``noun`` – the verb and the head noun of the target NP
  (rule-derived, with manual corrections in ``configs/gold_head_overrides.csv``);
* ``masked_context`` – the Chinese sentence only (interlinear glosses dropped),
  with the whole annotated target NP replaced by ``[TARGET_NP]``;
* the gold labels (form, qualia, typicality, context features) and the
  Part 1 cross-validation fold, which are never sent to the model.

A leakage audit flags items where ``一`` + classifier stays visible right
before the mask, or where the masked span does not start with ``一`` for a
YI_CL_N item.

The source sentences come from a copyrighted translation, so all outputs go
to ``data/private/`` (git-ignored).

Usage::

    python -m qualiaform_part2.gold.prepare \
        --gold data/private/part1/qualiaform_pilot.csv \
        --folds data/private/part1/split_assignments.csv \
        --out data/private/gold_v1
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
from typing import Any

from ..extract import digest, dump, write_rows

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_GOLD = ROOT / "data/private/part1/qualiaform_pilot.csv"
DEFAULT_FOLDS = ROOT / "data/private/part1/split_assignments.csv"
DEFAULT_OUT = ROOT / "data/private/gold_v1"
OVERRIDES = ROOT / "configs/gold_head_overrides.csv"
MASK = "[TARGET_NP]"
CJK = re.compile(r"[一-鿿]")
CONTEXT_FEATURES = ("negation", "distributivity", "neg_dis", "modifier", "aspect", "non_referential")


def read_csv(path: str | pathlib.Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def chinese_text(context: str) -> str:
    """Keep only lines containing Chinese characters (drops gloss lines); remove spaces."""
    lines = [line for line in context.splitlines() if CJK.search(line)]
    return re.sub(r"\s+", "", "".join(lines))


def mask_target(context: str, target_np: str) -> tuple[str, str, str]:
    """Replace the target NP with ``[TARGET_NP]``.

    Strategy: find the annotated NP string in the de-starred text, using the
    occurrence closest to the first ``*`` annotation mark. If the annotated
    NP is an abridged form not found verbatim, fall back to the starred span
    and extend it leftwards over a directly preceding ``一`` + classifier.

    Returns ``(masked_context, masked_span, method)``.
    """
    starred = chinese_text(context)
    star = starred.find("*")
    star_position = star - starred[:star].count("*") if star >= 0 else 0
    clean = starred.replace("*", "")
    target = re.sub(r"\s+", "", target_np)

    occurrences = [m.start() for m in re.finditer(re.escape(target), clean)] if target else []
    if occurrences:
        start = min(occurrences, key=lambda p: abs(p - star_position))
        end = start + len(target)
        method = "annotated_np"
    elif star >= 0:
        last = starred.rfind("*")
        start = star_position
        end = last - starred[:last].count("*")
        if start >= 2 and clean[start - 2] == "一":
            start -= 2
        method = "starred_span"
    else:
        raise ValueError("target NP cannot be located")
    return clean[:start] + MASK + clean[end:], clean[start:end], method


def align_to_yi(context: str, span: str, masked: str) -> tuple[str, str, str]:
    """For YI_CL_N items, make the mask start exactly at the numeral ``一``.

    * If the annotated span includes material before ``一`` (e.g. the verb
      in ``再给他一次机会``), that material is left visible.
    * If ``一`` + classifier precedes the span (e.g. ``穿一身猩红镶金[制服]``),
      the mask is extended leftwards, so the answer is not visible.

    Returns ``(masked_context, masked_span, adjustment)``.
    """
    start = masked.find(MASK)
    clean = masked[:start] + span + masked[start + len(MASK) :]
    end = start + len(span)
    if "一" in span and not span.startswith("一"):
        start += span.find("一")
        adjustment = "shrunk_to_yi"
    elif "一" not in span:
        window = clean[max(0, start - 8) : start]
        clause = re.split(r"[，。,；;！？!?、：:“”\"]", window)[-1]
        if "一" not in clause:
            return masked, span, "no_yi_found"
        start -= len(clause) - clause.rfind("一")
        adjustment = "extended_to_yi"
    else:
        return masked, span, "none"
    return clean[:start] + MASK + clean[end:], clean[start:end], adjustment


def derive_verb_noun(vn: str, target_np: str, form: str) -> tuple[str, str]:
    """Rule-based verb and head noun (corrected afterwards by the override table)."""
    vn = re.sub(r"\s+", "", vn)
    target = re.sub(r"\s+", "", target_np)
    position = vn.find(target)
    verb = vn[:position] if position > 0 else vn
    if position < 0 and "一" in vn:
        verb = vn[: vn.find("一")]
    verb = re.sub(r"(了|着|过)$", "", verb)
    noun = target
    if form == "YI_CL_N" and noun.startswith("一"):
        noun = noun[2:]
    if "的" in noun:
        noun = noun.split("的")[-1]
    return verb, noun


def leakage_flags(masked_context: str, masked_span: str, form: str) -> list[str]:
    """Audit flags; any flag means the item deserves a manual look."""
    flags = []
    before = masked_context[: masked_context.find(MASK)]
    if before[-2:-1] == "一" or before[-3:-2] == "一":
        flags.append("yi_visible_before_mask")
    if form == "YI_CL_N" and not masked_span.startswith("一"):
        flags.append("yi_not_in_masked_span")
    return flags


def build_items(gold_rows: list[dict], fold_rows: list[dict], overrides: dict[str, dict]) -> list[dict]:
    folds = {
        r["item_id"]: (int(r["fold"]), r["group_id"])
        for r in fold_rows
        if r["protocol"] == "grouped" and r["seed"] == "42"
    }
    items = []
    for row in gold_rows:
        item_id = row["item_id"]
        if not row["mandarin_context"].strip() or not row["mandarin_vn"].strip():
            continue  # no Chinese text to show a model (one item in Part 1)
        masked, span, method = mask_target(row["mandarin_context"], row["mandarin_np"])
        adjustment = "none"
        if row["form_target"] == "YI_CL_N":
            masked, span, adjustment = align_to_yi(row["mandarin_context"], span, masked)
        verb, noun = derive_verb_noun(row["mandarin_vn"], row["mandarin_np"], row["form_target"])
        head_source = "rule"
        if item_id in overrides:
            verb, noun = overrides[item_id]["verb"], overrides[item_id]["noun"]
            head_source = "manual_override"
        fold, group = folds[item_id]
        items.append(
            {
                "item_id": item_id,
                "fold": fold,
                "group_id": group,
                "chapter": int(row["chapter"]),
                # model input
                "verb": verb,
                "noun": noun,
                "masked_context": masked,
                # gold (never sent to the model)
                "form_target": row["form_target"],
                "qualia_gold": row["qualia_gold"],
                "typicality_gold": int(row["typicality_gold"]),
                **{f: int(row[f]) for f in CONTEXT_FEATURES},
                "qualia_note": row["qualia_notes"].split("。")[0] + "。",
                # provenance / audit
                "mandarin_vn": re.sub(r"\s+", "", row["mandarin_vn"]),
                "masked_span": span,
                "mask_method": method,
                "mask_adjustment": adjustment,
                "head_source": head_source,
                "leakage_flags": leakage_flags(masked, span, row["form_target"]),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gold-set LLM inputs.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new folder")
    overrides = {r["item_id"]: r for r in read_csv(OVERRIDES)}
    items = build_items(read_csv(args.gold), read_csv(args.folds), overrides)

    out.mkdir(parents=True)
    write_rows(out / "items.jsonl", items)
    summary: dict[str, Any] = {
        "items": len(items),
        "forms": dict(collections.Counter(i["form_target"] for i in items)),
        "qualia": dict(collections.Counter(i["qualia_gold"] for i in items)),
        "folds": dict(collections.Counter(i["fold"] for i in items)),
        "mask_method": dict(collections.Counter(i["mask_method"] for i in items)),
        "mask_adjustment": dict(collections.Counter(i["mask_adjustment"] for i in items)),
        "head_source": dict(collections.Counter(i["head_source"] for i in items)),
        "leakage_flags": dict(collections.Counter(f for i in items for f in i["leakage_flags"])),
        "gold_sha256": digest(pathlib.Path(args.gold).read_bytes()),
        "folds_sha256": digest(pathlib.Path(args.folds).read_bytes()),
        "overrides_sha256": digest(OVERRIDES.read_bytes()),
        "items_sha256": digest((out / "items.jsonl").read_bytes()),
    }
    dump(out / "manifest.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

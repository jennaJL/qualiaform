"""Extract verb–noun candidates from Universal Dependencies (UD) Chinese treebanks.

For every common noun that is a core or oblique argument of a verb, the
extractor decides which surface form the noun phrase (NP) has:

* ``BN``       – a bare noun: no numeral, determiner or classifier.
* ``YI_CL_N``  – exactly ``一`` + one classifier + noun, in that order.

Anything else (demonstratives, other numerals, nested determination,
discontinuous NPs) is excluded and counted in an audit log, so that the
extraction is conservative and every exclusion is accounted for.

The whole NP subtree is replaced by ``[TARGET_NP]`` in ``masked_context``.
Models only ever see this masked context, never the original surface form.

Labels are *observed surface forms* produced by rules. They are weak labels,
not expert judgements about indefiniteness.

Usage::

    python -m qualiaform_part2.extract --out data/processed/my_candidates
    python -m qualiaform_part2.extract --pilot --out data/my_pilot
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import platform
from collections.abc import Iterable, Iterator
from typing import Any

VERSION = "extract-v1.0.0"
LABELS = ("BN", "YI_CL_N")

CONLLU_FIELDS = ("id", "form", "lemma", "upos", "xpos", "feats", "head", "rel", "deps", "misc")
ARGUMENT_RELATIONS = ("obj", "nsubj", "obl", "iobj")
DETERMINATION_RELATIONS = ("det", "nummod", "clf")
FLAGGED_MODIFIER_RELATIONS = ("acl", "nmod", "conj", "amod", "appos")
MASK = "[TARGET_NP]"

Token = dict[str, Any]
Row = dict[str, Any]


# --------------------------------------------------------------------------
# Small I/O helpers shared by the whole package
# --------------------------------------------------------------------------


def digest(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def dump(path: str | pathlib.Path, obj: Any) -> None:
    """Write ``obj`` as pretty, key-sorted UTF-8 JSON (stable for hashing)."""
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    pathlib.Path(path).write_text(text, encoding="utf8")


def write_rows(path: str | pathlib.Path, rows: Iterable[Row]) -> None:
    """Write rows as JSON Lines with sorted keys."""
    lines = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    pathlib.Path(path).write_text("".join(lines), encoding="utf8")


def read_rows(path: str | pathlib.Path) -> list[Row]:
    """Read a JSON Lines file, skipping blank lines."""
    lines = pathlib.Path(path).read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --------------------------------------------------------------------------
# CoNLL-U parsing
# --------------------------------------------------------------------------


def _check_tree(tokens: list[Token]) -> None:
    """Raise ``ValueError`` if token IDs/heads are invalid or contain a cycle."""
    ids = {t["id"] for t in tokens}
    if len(ids) != len(tokens) or any(t["head"] not in ids | {0} for t in tokens):
        raise ValueError("Invalid token IDs/heads")
    by_id = {t["id"]: t for t in tokens}
    for token in tokens:
        seen: set[int] = set()
        current = token
        while current["head"]:
            if current["id"] in seen:
                raise ValueError("Dependency cycle")
            seen.add(current["id"])
            current = by_id[current["head"]]


def conllu(path: str | pathlib.Path) -> Iterator[tuple[dict[str, str], list[Token]]]:
    """Yield ``(metadata, tokens)`` per sentence of a CoNLL-U file.

    Multi-word token ranges (``1-2``) and empty nodes (``1.1``) are skipped.
    """
    meta: dict[str, str] = {}
    tokens: list[Token] = []
    lines = pathlib.Path(path).read_text(encoding="utf8").splitlines() + [""]
    for line in lines:
        if not line.strip():
            if tokens:
                _check_tree(tokens)
                yield meta, tokens
            meta, tokens = {}, []
        elif line.startswith("#"):
            if " = " in line:
                key, value = line[2:].split(" = ", 1)
                meta[key] = value
        else:
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError("Expected ten CoNLL-U columns")
            if "-" in fields[0] or "." in fields[0]:
                continue
            token = dict(zip(CONLLU_FIELDS, fields))
            token["id"] = int(fields[0])
            token["head"] = int(fields[6])
            tokens.append(token)


# --------------------------------------------------------------------------
# Candidate extraction
# --------------------------------------------------------------------------


def _base_relation(token: Token) -> str:
    """``obl:tmod`` -> ``obl``."""
    return token["rel"].split(":")[0]


def _subtree(root_id: int, children: dict[int, list[Token]]) -> set[int]:
    """Return the IDs of ``root_id`` and all of its descendants."""
    subtree: set[int] = set()
    stack = [root_id]
    while stack:
        node = stack.pop()
        subtree.add(node)
        stack.extend(child["id"] for child in children[node])
    return subtree


def _classify_form(noun: Token, children: dict[int, list[Token]]) -> tuple[str | None, list[Token]]:
    """Decide the surface form of the NP headed by ``noun``.

    Returns ``(label, classifiers)``; ``label`` is ``None`` when the NP has
    some other kind of determination and must be excluded.
    """
    dependents = children[noun["id"]]
    numerals = [t for t in dependents if t["rel"] == "nummod"]
    determiners = [t for t in dependents if t["rel"] == "det"]
    has_classifier = any(t["rel"] == "clf" for t in dependents)

    yi = numerals[0] if len(numerals) == 1 and numerals[0]["form"] == "一" else None
    classifiers: list[Token] = []
    if yi:
        # UD attaches the classifier either to the numeral or to the noun.
        classifiers = [t for t in children[yi["id"]] if t["rel"] == "clf"]
        classifiers += [t for t in dependents if t["rel"] == "clf"]

    if not (numerals or determiners or has_classifier):
        return "BN", classifiers
    is_yi_cl_n = (
        yi is not None
        and len(classifiers) == 1
        and not determiners
        and yi["id"] < classifiers[0]["id"] < noun["id"]
    )
    return ("YI_CL_N", classifiers) if is_yi_cl_n else (None, classifiers)


def extract(path: str | pathlib.Path, split: str, audit: collections.Counter | None = None) -> Iterator[Row]:
    """Yield one candidate row per eligible verb–noun pair in ``path``.

    ``audit`` (a Counter) is updated with how many nouns were seen,
    accepted and excluded, and why.
    """
    audit = audit if audit is not None else collections.Counter()
    for index, (meta, tokens) in enumerate(conllu(path), 1):
        audit["sentences"] += 1
        by_id = {t["id"]: t for t in tokens}
        children: dict[int, list[Token]] = collections.defaultdict(list)
        for token in tokens:
            children[token["head"]].append(token)

        for noun in tokens:
            if noun["upos"] != "NOUN" or noun["xpos"] == "NNB":
                continue
            audit["nouns"] += 1

            verb = by_id.get(noun["head"])
            if not verb or verb["upos"] != "VERB" or _base_relation(noun) not in ARGUMENT_RELATIONS:
                audit["excluded_non_verbal_argument"] += 1
                continue

            label, classifiers = _classify_form(noun, children)
            if label is None:
                audit["excluded_determination"] += 1
                continue

            # The whole argument subtree is removed from the model context,
            # including every surface marker of the form.
            subtree = _subtree(noun["id"], children)
            if label == "BN" and any(
                _base_relation(by_id[i]) in DETERMINATION_RELATIONS or by_id[i]["upos"] == "NUM"
                for i in subtree
            ):
                audit["excluded_nested_determination"] += 1
                continue

            first, last = min(subtree), max(subtree)
            if subtree != set(range(first, last + 1)):
                audit["excluded_discontinuous_np"] += 1
                continue
            if verb["id"] in subtree:
                raise ValueError("Predicate inside nominal subtree")

            row = _candidate_row(
                meta, tokens, index, split, noun, verb, label, classifiers, subtree, children, by_id
            )
            audit["accepted_" + label] += 1
            yield row


def _candidate_row(
    meta: dict[str, str],
    tokens: list[Token],
    index: int,
    split: str,
    noun: Token,
    verb: Token,
    label: str,
    classifiers: list[Token],
    subtree: set[int],
    children: dict[int, list[Token]],
    by_id: dict[int, Token],
) -> Row:
    """Build the output record for one accepted candidate.

    Key order matters: it fixes the column order of ``candidates.csv``.
    """
    sentence = meta.get("text", "".join(t["form"] for t in tokens))
    sentence_id = meta.get("sent_id", f"{split}-{index}")
    first = min(subtree)
    masked_context = "".join(
        MASK if t["id"] == first else "" if t["id"] in subtree else t["form"] for t in tokens
    )

    flags = [
        "has_" + rel
        for rel in FLAGGED_MODIFIER_RELATIONS
        if any(_base_relation(t) == rel for t in children[noun["id"]])
    ]
    if any(by_id[i]["upos"] == "PROPN" for i in subtree):
        flags.append("contains_proper_name")

    return {
        "candidate_id": f"{split}:{sentence_id}:{noun['id']}",
        "source_split": split,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "sentence_hash": digest(sentence.encode()),
        "noun": noun["form"],
        "noun_lemma": noun["lemma"],
        "verb": verb["form"],
        "verb_lemma": verb["lemma"],
        "noun_token_id": noun["id"],
        "verb_token_id": verb["id"],
        "relation": noun["rel"],
        "observed_form": label,
        "classifier": classifiers[0]["form"] if label == "YI_CL_N" else "",
        "np_token_ids": sorted(subtree),
        "np_text": "".join(t["form"] for t in tokens if t["id"] in subtree),
        "masked_context": masked_context,
        "flags": flags,
        "indefinite_status": "unreviewed",
        "gold_typicality": None,
        "gold_qualia": None,
        "extractor_version": VERSION,
        "dependency_info": tokens,
    }


# --------------------------------------------------------------------------
# Snapshot writing
# --------------------------------------------------------------------------


def _statistics(rows: list[Row], audits: dict[str, dict]) -> dict[str, Any]:
    """Summary counts, duplicate checks and train/dev/test overlap checks."""
    hashes_by_split = {
        split: {r["sentence_hash"] for r in rows if r["source_split"] == split}
        for split in ("train", "dev", "test")
    }
    lengths = sorted(len(r["dependency_info"]) for r in rows)
    unique_sentences = len({r["sentence_hash"] for r in rows})
    split_pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    return {
        "total": len(rows),
        "by_form": dict(collections.Counter(r["observed_form"] for r in rows)),
        "by_split_form": dict(
            collections.Counter(r["source_split"] + "/" + r["observed_form"] for r in rows)
        ),
        "unique_nouns": len({r["noun_lemma"] for r in rows}),
        "unique_sentences": unique_sentences,
        "same_sentence_extra_occurrences": len(rows) - unique_sentences,
        "exact_occurrence_duplicates": len(rows)
        - len({(r["sentence_hash"], r["noun_token_id"]) for r in rows}),
        "cross_split_sentence_overlap": {
            f"{a}/{b}": len(hashes_by_split[a] & hashes_by_split[b]) for a, b in split_pairs
        },
        "sentence_token_length": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "median": lengths[len(lengths) // 2] if lengths else 0,
        },
        "flags": dict(collections.Counter(flag for r in rows for flag in r["flags"])),
        "audit": audits,
    }


def _write_csv(path: pathlib.Path, rows: list[Row]) -> None:
    """Spreadsheet-friendly copy without the bulky dependency trace."""
    fields = [k for k in rows[0] if k != "dependency_info"] if rows else []
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list) else row[k]
                    for k in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/ud", help="folder with zh_gsdsimp-ud-*.conllu")
    parser.add_argument("--out", required=True, help="new output folder (never overwritten)")
    parser.add_argument("--pilot", action="store_true", help="40-row dev pilot, minority first")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("Refusing to overwrite snapshot; choose a new directory.")

    audits: dict[str, dict] = {}
    rows: list[Row] = []
    sources: dict[str, str] = {}
    for split in ("dev",) if args.pilot else ("train", "dev", "test"):
        path = pathlib.Path(args.data) / f"zh_gsdsimp-ud-{split}.conllu"
        audit: collections.Counter = collections.Counter()
        rows += list(extract(path, split, audit))
        audits[split] = dict(audit)
        sources[path.name] = digest(path.read_bytes())

    if args.pilot:
        # Review every available minority example before filling the pilot with BN.
        rows = sorted(rows, key=lambda r: (r["observed_form"] == "BN", r["candidate_id"]))[:40]
    rows.sort(key=lambda r: r["candidate_id"])
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs")

    out.mkdir(parents=True)
    write_rows(out / "candidates.jsonl", rows)
    _write_csv(out / "candidates.csv", rows)
    stats = _statistics(rows, audits)
    dump(out / "statistics.json", stats)
    dump(
        out / "manifest.json",
        {
            "version": VERSION,
            "created_date": "2026-10-07",
            "python": platform.python_version(),
            "input_sha256": sources,
            "output_sha256": {f.name: digest(f.read_bytes()) for f in sorted(out.iterdir())},
            "label_status": "observed form, not indefinite gold",
            "sampling": (
                "minority-first dev pilot, 40"
                if args.pilot
                else "all eligible occurrences; no class rebalance"
            ),
            "code_sha256": digest(pathlib.Path(__file__).read_bytes()),
            "corpus_release": "unknown; exact uploaded bytes preserved",
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

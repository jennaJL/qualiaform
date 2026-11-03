"""Supervised baselines on the UD candidates: frozen encoder + PyTorch heads.

Pipeline:

1. ``features`` – encode every candidate's masked input with a pinned
   multilingual MiniLM (Hugging Face Transformers), mean-pool and
   L2-normalise to 384-d vectors. The encoder is frozen.
2. ``train`` – on the train split, train a linear head and a one-hidden-layer
   MLP head for each of three seeds (PyTorch), plus a character TF-IDF +
   logistic regression baseline (scikit-learn).

Evaluation protocol:

* Standardisation statistics and vocabularies are fit on train only.
* Each model's decision threshold is chosen on dev (macro-F1).
* The served model is chosen on dev using the primary seed only.
* Test is scored once, after selection. Note that this test split was
  already used in the earlier LLM experiments (it is a reused benchmark).

Usage::

    python -m qualiaform_part2.portfolio features --out data/my_features
    python -m qualiaform_part2.portfolio train --features data/portfolio_features_v1 \
        --out results/my_run
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import time
from typing import Any

import numpy as np

from .evaluate import metrics
from .extract import LABELS, dump, read_rows, write_rows

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/portfolio_protocol_v1.json"
DATASET = ROOT / "data/processed/ud_candidates_v1/candidates.jsonl"
ENCODER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
]
BN, YI = 0, 1  # class indices; LABELS[BN] == "BN", LABELS[YI] == "YI_CL_N"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text())


def sha_file(path: str | pathlib.Path) -> str:
    """SHA-256 of a (possibly large) file, read in 8 MB chunks."""
    hasher = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def input_text(row: dict[str, Any]) -> str:
    """The single text string the encoder sees (no surface-form information)."""
    return (
        f"名词：{row['noun_lemma']}；动词：{row['verb_lemma']}；"
        f"关系：{row['relation']}；上下文：{row['masked_context']}"
    )


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


def encoder_path(override: str | None = None) -> pathlib.Path:
    """Find the pinned MiniLM weights: argument, env var, ./models, or HF cache."""
    cfg = load_protocol()
    cache_name = "models--" + cfg["encoder_repo"].replace("/", "--")
    options = [
        override,
        os.environ.get("QUALIAFORM_ENCODER"),
        ROOT / "models/minilm_base",
        pathlib.Path.home() / ".cache/huggingface/hub" / cache_name / "snapshots" / cfg["encoder_revision"],
    ]
    for option in options:
        if option and (pathlib.Path(option) / "model.safetensors").exists():
            return pathlib.Path(option).resolve()
    raise FileNotFoundError(
        "Pinned MiniLM weights not found; run scripts/setup_portfolio.py or set QUALIAFORM_ENCODER"
    )


class Encoder:
    """Frozen sentence encoder: mean pooling over tokens, then L2 normalisation."""

    def __init__(self, path: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(4)
        self.path = encoder_path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            str(self.path), local_files_only=True, trust_remote_code=False
        ).eval()
        self.max_length = load_protocol()["max_length"]

    def encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        torch = self.torch
        texts = [input_text(r) for r in rows]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()


def features(args: argparse.Namespace) -> None:
    """Encode all candidates once and save a checksummed feature snapshot."""
    cfg = load_protocol()
    out = pathlib.Path(args.out)
    if out.exists():
        raise ValueError("Feature snapshot exists; use a new directory")
    out.mkdir(parents=True)

    rows = read_rows(DATASET)
    encoder = Encoder(args.encoder)
    batch_size = cfg["encoder_batch_size"]
    start = time.monotonic()
    chunks = []
    for i in range(0, len(rows), batch_size):
        chunks.append(encoder.encode(rows[i : i + batch_size]))
        if i % 320 == 0:
            print(f"Encoded {min(i + 32, len(rows))}/{len(rows)}", flush=True)
    x = np.concatenate(chunks)
    np.savez_compressed(out / "features.npz", x=x, ids=np.array([r["candidate_id"] for r in rows]))

    encoder_manifest = {
        "repo": cfg["encoder_repo"],
        "revision": cfg["encoder_revision"],
        "files": {
            name: {"sha256": sha_file(encoder.path / name), "bytes": (encoder.path / name).stat().st_size}
            for name in ENCODER_FILES
            if (encoder.path / name).is_file()
        },
    }
    dump(ROOT / "configs/encoder_manifest.json", encoder_manifest)

    lengths = [len(encoder.tokenizer.encode(input_text(r))) for r in rows]
    dump(
        out / "manifest.json",
        {
            "dataset_sha256": sha_file(DATASET),
            "protocol_sha256": sha_file(PROTOCOL),
            "encoder_manifest_sha256": sha_file(ROOT / "configs/encoder_manifest.json"),
            "features_sha256": sha_file(out / "features.npz"),
            "count": len(rows),
            "dimensions": x.shape[1],
            "truncated_inputs": sum(n > cfg["max_length"] for n in lengths),
            "elapsed_seconds": time.monotonic() - start,
            "label_fields_used": False,
        },
    )
    print("FEATURES COMPLETE", x.shape, flush=True)


# --------------------------------------------------------------------------
# Classifier heads, thresholds and scoring
# --------------------------------------------------------------------------


def head(kind: str, dim: int, hidden: int = 128, dropout: float = 0.1):
    """A linear head, or an MLP head with one hidden ReLU layer."""
    import torch.nn as nn

    if kind == "linear":
        return nn.Linear(dim, 2)
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))


def _labelled_pairs(y: np.ndarray, scores: np.ndarray, threshold: float) -> list[tuple[str, str]]:
    return [(LABELS[int(ref)], LABELS[int(s >= threshold)]) for ref, s in zip(y, scores)]


def choose_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Pick the YI threshold that maximises dev macro-F1.

    Ties are broken by balanced accuracy, then by closeness to 0.5.
    Returns ``(threshold, macro_f1)``.
    """
    candidates = []
    for t in np.linspace(0.005, 0.995, 199):
        m = metrics(_labelled_pairs(y, probs, t))
        candidates.append((m["macro_f1"], m["balanced_accuracy"] or 0, -abs(float(t) - 0.5), float(t)))
    best = max(candidates)
    return best[-1], best[0]


def score(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    return metrics(_labelled_pairs(y, prob, threshold))


def bootstrap(
    rows: list[dict[str, Any]], prob: np.ndarray, threshold: float, repeats: int = 500
) -> dict[str, Any]:
    """95% intervals from resampling whole sentences (not single candidates).

    Candidates from the same sentence are correlated, so the sentence is
    the resampling unit.
    """
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["sentence_hash"]].append(i)
    clusters = list(groups.values())
    rng = np.random.default_rng(1729)

    values = []
    for _ in range(repeats):
        sample = [i for j in rng.integers(0, len(clusters), len(clusters)) for i in clusters[j]]
        m = metrics([(rows[i]["observed_form"], LABELS[int(prob[i] >= threshold)]) for i in sample])
        if m["balanced_accuracy"] is not None:
            values.append([m["balanced_accuracy"], m["macro_f1"]])
    q = np.quantile(np.array(values), [0.025, 0.975], axis=0)
    return {
        "resampling_unit": "sentence",
        "resamples": len(values),
        "balanced_accuracy_95": [float(q[0, 0]), float(q[1, 0])],
        "macro_f1_95": [float(q[0, 1]), float(q[1, 1])],
        "interpretation": "descriptive uncertainty on this reused weak-label benchmark; "
        "not significance evidence",
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _load_data(feature_dir: pathlib.Path):
    """Load and integrity-check features, rows, labels and split indices."""
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    if manifest["dataset_sha256"] != sha_file(DATASET) or manifest["protocol_sha256"] != sha_file(PROTOCOL):
        raise ValueError("Feature snapshot does not match dataset/protocol")
    if manifest["features_sha256"] != sha_file(feature_dir / "features.npz"):
        raise ValueError("Feature file checksum mismatch")

    data = np.load(feature_dir / "features.npz", allow_pickle=False)
    rows = read_rows(DATASET)
    if list(data["ids"]) != [r["candidate_id"] for r in rows]:
        raise ValueError("Feature/row ID mismatch")

    x = data["x"]
    y = np.array([LABELS.index(r["observed_form"]) for r in rows])
    split_index = {
        split: np.array([i for i, r in enumerate(rows) if r["source_split"] == split])
        for split in ("train", "dev", "test")
    }
    train_sentences = {rows[i]["sentence_hash"] for i in split_index["train"]}
    for split in ("dev", "test"):
        if train_sentences & {rows[i]["sentence_hash"] for i in split_index[split]}:
            raise ValueError(f"train/{split} sentence overlap")
    return x, y, rows, split_index


def _train_head(torch, kind, seed, z, yt, train_ix, class_weights, cfg, history):
    """Train one classification head with AdamW and weighted cross-entropy."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = head(kind, z.shape[1], cfg["mlp_hidden"], cfg["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(
        TensorDataset(z[train_ix], yt[train_ix]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(yb)
            seen += len(yb)
        history.append({"model": kind, "seed": seed, "epoch": epoch + 1, "train_loss": total_loss / seen})
    model.eval()
    return model


def _lexical_baseline(rows, y, ix, cfg, out):
    """Character 1–3-gram TF-IDF + class-balanced logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    text = [input_text(r) for r in rows]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 3), min_df=2, max_features=20000, sublinear_tf=True
    )
    train_x = vectorizer.fit_transform([text[i] for i in ix["train"]])
    model = LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=1729)
    model.fit(train_x, y[ix["train"]])

    dev_probs = model.predict_proba(vectorizer.transform([text[i] for i in ix["dev"]]))[:, 1]
    threshold, _ = choose_threshold(y[ix["dev"]], dev_probs)
    dump(
        out / "lexical_model.json",
        {
            "threshold": threshold,
            "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
            "dev": score(y[ix["dev"]], dev_probs, threshold),
            "configuration": cfg["lexical_baseline"],
        },
    )
    np.savez_compressed(
        out / "lexical_weights.npz", idf=vectorizer.idf_, coef=model.coef_, intercept=model.intercept_
    )
    test_probs = model.predict_proba(vectorizer.transform([text[i] for i in ix["test"]]))[:, 1]
    return test_probs, threshold


def _prediction_rows(test_rows, probs, threshold):
    return [
        {
            "candidate_id": r["candidate_id"],
            "observed_form": r["observed_form"],
            "prediction": LABELS[int(p >= threshold)],
            "yi_score_uncalibrated": float(p),
        }
        for r, p in zip(test_rows, probs)
    ]


def train(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    cfg = load_protocol()
    out = pathlib.Path(args.out)
    if out.exists():
        raise ValueError("Benchmark snapshot exists; use a new directory")
    feature_dir = pathlib.Path(args.features)
    x, y, rows, ix = _load_data(feature_dir)

    # Standardise with train statistics only.
    mean = x[ix["train"]].mean(0)
    std = x[ix["train"]].std(0).clip(0.01)
    z = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)

    out.mkdir(parents=True)
    dump(out / "protocol.json", cfg)
    # Square-root inverse-frequency class weights soften the 35:1 imbalance.
    counts = np.bincount(y[ix["train"]], minlength=2)
    class_weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32)

    history: list[dict[str, Any]] = []
    models: list[tuple[str, Any, dict[str, Any]]] = []
    start = time.monotonic()
    for kind in cfg["models"]:
        for seed in cfg["seeds"]:
            model = _train_head(torch, kind, seed, z, yt, ix["train"], class_weights, cfg, history)
            with torch.inference_mode():
                dev_probs = model(z[ix["dev"]]).softmax(-1)[:, 1].numpy()
            threshold, dev_f1 = choose_threshold(y[ix["dev"]], dev_probs)

            tag = f"{kind}_{seed}"
            folder = out / tag
            folder.mkdir()
            state = {k: v.contiguous() for k, v in model.state_dict().items()}
            state["feature_mean"] = torch.from_numpy(mean)
            state["feature_std"] = torch.from_numpy(std)
            save_file(state, str(folder / "model.safetensors"))
            meta = {
                "kind": kind,
                "seed": seed,
                "input_dim": z.shape[1],
                "hidden": cfg["mlp_hidden"],
                "dropout": cfg["dropout"],
                "threshold": threshold,
                "score_interpretation": "uncalibrated YI score",
                "dev": score(y[ix["dev"]], dev_probs, threshold),
                "parameters": sum(t.numel() for t in model.parameters()),
                "encoder_frozen": True,
                "weights_sha256": sha_file(folder / "model.safetensors"),
            }
            dump(folder / "model.json", meta)
            models.append((tag, model, meta))
            print(tag, "dev_macro_f1", dev_f1, "threshold", threshold, flush=True)

    # The served model is fixed using dev only, before any test scoring.
    primary_seed = cfg["seeds"][0]
    eligible = [m for m in models if m[2]["seed"] == primary_seed]
    champion_tag, _, champion_meta = max(eligible, key=lambda m: m[2]["dev"]["macro_f1"])
    dump(
        out / "selection.json",
        {
            "selected": champion_tag,
            "criterion": "dev macro-F1, primary seed 1729; test not consulted",
            "dev_macro_f1": champion_meta["dev"]["macro_f1"],
        },
    )

    lexical_probs, lexical_threshold = _lexical_baseline(rows, y, ix, cfg, out)

    # ---- test scoring (once) ----
    test_rows = [rows[i] for i in ix["test"]]
    y_test = y[ix["test"]]
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for tag, model, meta in models:
            probs = model(z[ix["test"]]).softmax(-1)[:, 1].numpy()
            results[tag] = score(y_test, probs, meta["threshold"])
            write_rows(
                out / tag / "test_predictions.jsonl", _prediction_rows(test_rows, probs, meta["threshold"])
            )
            if tag == champion_tag:
                selected_probs = probs
    results["char_tfidf_logreg"] = score(y_test, lexical_probs, lexical_threshold)
    results["majority_BN"] = score(y_test, np.zeros(len(test_rows)), 0.5)
    write_rows(
        out / "lexical_test_predictions.jsonl",
        _prediction_rows(test_rows, lexical_probs, lexical_threshold),
    )
    dump(out / "test_metrics.json", results)
    write_rows(out / "training_history.jsonl", history)

    threshold = champion_meta["threshold"]
    dump(out / "selected_bootstrap.json", bootstrap(test_rows, selected_probs, threshold))
    _write_slices_and_errors(out, rows, ix, test_rows, y_test, selected_probs, threshold)

    dump(
        out / "manifest.json",
        {
            "status": "COMPLETED",
            "created_date": "2026-10-08",
            "dataset_sha256": sha_file(DATASET),
            "feature_sha256": sha_file(feature_dir / "features.npz"),
            "protocol_sha256": sha_file(PROTOCOL),
            "code_sha256": sha_file(__file__),
            "train_n": len(ix["train"]),
            "dev_n": len(ix["dev"]),
            "test_n": len(ix["test"]),
            "elapsed_seconds": time.monotonic() - start,
            "selected": champion_tag,
            "gradient_updates": "classification heads only; encoder frozen",
            "test_reuse_notice": cfg["test_policy"],
            "reference": "weak observed-form labels; not expert indefinite gold",
            "paid_api_calls": 0,
        },
    )
    print("BENCHMARK COMPLETE", champion_tag, results[champion_tag], flush=True)


def _write_slices_and_errors(out, rows, ix, test_rows, y_test, probs, threshold):
    """Slice scores (seen vs unseen verb–noun pairs, proper names) and error list."""
    train_pairs = {(rows[i]["verb_lemma"], rows[i]["noun_lemma"]) for i in ix["train"]}
    masks = {
        "seen_VN": [(r["verb_lemma"], r["noun_lemma"]) in train_pairs for r in test_rows],
        "unseen_VN": [(r["verb_lemma"], r["noun_lemma"]) not in train_pairs for r in test_rows],
        "contains_proper_name": ["contains_proper_name" in r["flags"] for r in test_rows],
    }
    slices = {}
    for name, mask in masks.items():
        idx = np.where(mask)[0]
        slices[name] = score(y_test[idx], probs[idx], threshold)
    dump(out / "selected_slices.json", slices)

    errors = []
    for row, p in zip(test_rows, probs):
        predicted = LABELS[int(p >= threshold)]
        if row["observed_form"] != predicted:
            errors.append(
                {
                    **row,
                    "prediction": predicted,
                    "yi_score_uncalibrated": float(p),
                    "error_type": "unreviewed",
                }
            )
    write_rows(out / "errors.jsonl", errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-encoder features and PyTorch heads.")
    parser.add_argument("command", choices=["features", "train"])
    parser.add_argument("--encoder", help="path to pinned MiniLM weights")
    parser.add_argument("--features", default=str(ROOT / "data/portfolio_features_v1"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "features":
        features(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

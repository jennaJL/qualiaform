"""Offline LLM inference and LoRA fine-tuning on CPU (Qwen2.5-0.5B-Instruct).

No API keys or paid services are involved.

``infer``
    Runs one prompt condition (``direct`` or ``structured``; ``zero`` or
    ``corpus_few`` shots) over a split. Output is *constrained* to a finite
    set of valid JSON answers: a token trie built from every allowed answer
    restricts generation, and the model's own logits choose among the
    allowed branches. Runs can be resumed after interruption.

``train``
    Parameter-efficient fine-tuning (LoRA on q_proj/v_proj of the last four
    layers) on a balanced 256-example train sample. Loss is computed on the
    assistant answer tokens only.

Usage::

    python -m qualiaform_part2.local_cpu infer --model models/base_qwen05 \
        --condition direct --shots zero --split test --out results/my_run
    python -m qualiaform_part2.local_cpu train --model models/base_qwen05 --out models/my_lora
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import time
from typing import Any

from .experiment import QUALIA_VALUES, TYPICALITY_VALUES, model_input, validate_response
from .extract import LABELS, digest, dump, read_rows, write_rows

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/processed/ud_candidates_v1/candidates.jsonl"
LLM_DATA = ROOT / "data/local_llm_v1"
IGNORE_INDEX = -100  # label value ignored by PyTorch cross-entropy


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Prompts and requests
# --------------------------------------------------------------------------


def messages(request: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a stored request into chat messages (system, demos, input)."""
    chat = [{"role": "system", "content": request["system"]}]
    for demo in request["demonstrations"]:
        chat.append({"role": "user", "content": json.dumps(demo["input"], ensure_ascii=False)})
        chat.append({"role": "assistant", "content": json.dumps(demo["output"], ensure_ascii=False)})
    chat.append({"role": "user", "content": json.dumps(request["input"], ensure_ascii=False)})
    return chat


def requests_for(rows: list[dict[str, Any]], condition: str, shots: str) -> list[dict[str, Any]]:
    """Build hashed requests; ``corpus_few`` adds four train-split demos.

    Raises ``ValueError`` if any demo shares a sentence or masked context
    with an evaluation row.
    """
    prompt = (ROOT / f"prompts/local_v2/{condition}.txt").read_text()
    demos = []
    if shots == "corpus_few":
        for demo in read_rows(LLM_DATA / "corpus_demos.jsonl"):
            if demo["source_split"] != "train":
                raise ValueError("non-train demonstration")
            if any(
                demo["sentence_hash"] == r["sentence_hash"]
                or demo["input"]["masked_context"] == r["masked_context"]
                for r in rows
            ):
                raise ValueError("demo/evaluation overlap")
            output = (
                demo["output"]
                if condition == "structured"
                else {"form_prediction": demo["output"]["form_prediction"]}
            )
            demos.append({"input": demo["input"], "output": output})

    generation = {"do_sample": False, "seed": 1729, "max_new_tokens": 64, "decoder": "finite_json_enum_v1"}
    result = []
    for row in rows:
        request = {
            "system": prompt,
            "demonstrations": demos,
            "input": model_input(row),
            "generation": generation,
        }
        request_hash = digest(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
        result.append(
            {"candidate_id": row["candidate_id"], "request_sha256": request_hash, "request": request}
        )
    return result


def parse_response(raw: str, condition: str) -> dict[str, Any]:
    """Parse model output (optionally wrapped in a ```json fence) and validate it."""
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    return validate_response(json.loads(text), condition)


# --------------------------------------------------------------------------
# Constrained decoding
# --------------------------------------------------------------------------


def response_candidates(condition: str) -> list[dict[str, Any]]:
    """Every valid answer object: 2 for direct, 3 x 4 x 2 = 24 for structured."""
    if condition == "direct":
        return [{"form_prediction": label} for label in LABELS]
    return [
        {"typicality": t, "qualia": q, "form_prediction": label, "evidence": ""}
        for t in TYPICALITY_VALUES
        for q in QUALIA_VALUES
        for label in LABELS
    ]


def constrained_decoder(tokenizer, condition: str, prefix_length: int):
    """Return a ``prefix_allowed_tokens_fn`` for ``model.generate``.

    A trie of the token sequences of all valid answers is built. At each
    step only children of the current trie node are allowed, so output is
    always valid JSON with valid enum values. The *model's* logits pick the
    branch; no reference labels are available here.
    """
    trie: dict[int, dict] = {}
    for obj in response_candidates(condition):
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        node = trie
        for token in tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]:
            node = node.setdefault(token, {})

    def allowed(batch_id: int, input_ids) -> list[int]:
        node = trie
        for token in input_ids.tolist()[prefix_length:]:
            if token == tokenizer.eos_token_id:
                return [tokenizer.eos_token_id]
            node = node[token]
        return list(node) or [tokenizer.eos_token_id]

    return allowed


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def load_runtime(path: str, adapter: str | None = None):
    """Load tokenizer and float32 model from local files (optionally + LoRA adapter).

    Imports are local so the rest of the package works without PyTorch.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    return torch, model, tokenizer


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def _run_identity(args, rows, requests, model_manifest) -> dict[str, Any]:
    """Everything that defines a run; resuming requires an exact match."""
    adapter = pathlib.Path(args.adapter) if args.adapter else None
    return {
        "condition": args.condition,
        "shots": args.shots,
        "split": args.split,
        "count": len(rows),
        "batch_size": args.batch_size,
        "model_revision": model_manifest["revision"],
        "adapter": str(adapter.resolve()) if adapter else None,
        "adapter_sha256": digest((adapter / "adapter_model.safetensors").read_bytes()) if adapter else None,
        "dataset_sha256": digest(DATASET.read_bytes()),
        "requests_sha256": digest(json.dumps(requests, ensure_ascii=False, sort_keys=True).encode()),
        "code_sha256": digest(pathlib.Path(__file__).read_bytes()),
    }


def _completed_predictions(out: pathlib.Path, identity: dict, requests: list[dict]) -> list[dict]:
    """Return already-written predictions after checking they form a valid prefix."""
    if not (out / "identity.json").exists():
        dump(out / "identity.json", identity)
        write_rows(out / "requests.jsonl", requests)
        return []
    if json.loads((out / "identity.json").read_text()) != identity:
        raise ValueError("Resume identity mismatch")
    previous = read_rows(out / "predictions.jsonl") if (out / "predictions.jsonl").exists() else []
    if len(previous) != len({p["candidate_id"] for p in previous}):
        raise ValueError("Duplicate predictions")
    if [p["candidate_id"] for p in previous] != [r["candidate_id"] for r in requests[: len(previous)]]:
        raise ValueError("Non-prefix partial run")
    return previous


def infer(args: argparse.Namespace) -> None:
    if args.split == "pilot":
        rows = read_rows(LLM_DATA / "pilot_rows.jsonl")
    else:
        rows = [r for r in read_rows(DATASET) if r["source_split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    requests = requests_for(rows, args.condition, args.shots)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_manifest = json.loads((ROOT / "configs/cpu_model_manifest.json").read_text())
    identity = _run_identity(args, rows, requests, model_manifest)
    done = len(_completed_predictions(out, identity, requests))
    if done == len(rows) and (out / "manifest.json").exists():
        print("Already complete", out)
        return

    torch, model, tokenizer = load_runtime(args.model, args.adapter)
    start = time.monotonic()
    with (out / "predictions.jsonl").open("a", encoding="utf8") as handle:
        for offset in range(done, len(requests), args.batch_size):
            batch = requests[offset : offset + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    messages(r["request"]), tokenize=False, add_generation_prompt=True
                )
                for r in batch
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            prompt_length = inputs["input_ids"].shape[1]
            batch_start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=batch[0]["request"]["generation"]["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=constrained_decoder(tokenizer, args.condition, prompt_length),
                )
            outputs = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

            for request, raw in zip(batch, outputs):
                try:
                    parsed, error = parse_response(raw, args.condition), None
                except (ValueError, TypeError) as exc:
                    parsed, error = None, str(exc)
                record = {
                    "candidate_id": request["candidate_id"],
                    "prediction": parsed,
                    "raw_response": raw,
                    "error": error,
                    "request_sha256": request["request_sha256"],
                    # "external" is kept for compatibility with evaluate.py; backend is local.
                    "execution_mode": "external",
                    "backend": "local_cpu",
                    "generated_at": now(),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            # Flush every batch so an interrupted run can resume safely.
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"{args.condition}/{args.shots} {offset + len(batch)}/{len(requests)} "
                f"batch_seconds={time.monotonic() - batch_start:.1f}",
                flush=True,
            )

    few_shot = args.shots == "corpus_few"
    dump(
        out / "manifest.json",
        {
            "execution_mode": "external",
            "scientific_model_experiment_run": True,
            "backend": "local_cpu",
            "paid_api_calls": 0,
            "model_id": model_manifest["repo"],
            "model_revision": model_manifest["revision"],
            "config": identity,
            "config_sha256": digest(json.dumps(identity, sort_keys=True).encode()),
            "dataset_sha256": identity["dataset_sha256"],
            "count": len(rows),
            "prompt_sha256": digest(requests[0]["request"]["system"].encode()),
            "demonstration_sha256": (
                digest((LLM_DATA / "corpus_demos.jsonl").read_bytes()) if few_shot else None
            ),
            "semantic_demo_provenance": (
                "LLM-assisted provisional annotations on UD train, not Part1 gold" if few_shot else None
            ),
            "elapsed_this_invocation_seconds": time.monotonic() - start,
            "completed_at": now(),
            "files": {p.name: digest(p.read_bytes()) for p in out.iterdir() if p.is_file()},
        },
    )


# --------------------------------------------------------------------------
# LoRA training
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    import random

    from peft import LoraConfig, TaskType, get_peft_model

    out = pathlib.Path(args.out)
    if out.exists():
        raise ValueError("Refusing to overwrite training")
    protocol = json.loads((ROOT / "configs/local_protocol_v2.json").read_text())
    cfg = protocol["training"]

    torch, base, tokenizer = load_runtime(args.model)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        layers_to_transform=cfg["layers"],
        bias="none",
    )
    model = get_peft_model(base, lora)
    model.config.use_cache = False
    prompt = (ROOT / "prompts/local_v2/direct.txt").read_text()

    def encode(row):
        """Token IDs for prompt + answer; labels mask everything but the answer."""
        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(model_input(row), ensure_ascii=False)},
        ]
        answer = {"role": "assistant", "content": json.dumps({"form_prediction": row["observed_form"]})}
        prefix = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(chat + [answer], tokenize=True)
        assert full[: len(prefix)] == prefix
        if len(full) > cfg["max_length"]:
            raise ValueError("Training sample exceeds frozen length limit")
        return full, [IGNORE_INDEX] * len(prefix) + full[len(prefix) :]

    train_rows = read_rows(LLM_DATA / "train_rows.jsonl")
    valid_rows = read_rows(LLM_DATA / "dev_rows.jsonl")
    assert all(r["source_split"] == "train" for r in train_rows)
    assert all(r["source_split"] == "dev" for r in valid_rows)
    train_set = [encode(r) for r in train_rows]
    valid_set = [encode(r) for r in valid_rows]

    def collate(data):
        """Left-pad a list of (ids, labels) pairs into tensors."""
        length = max(len(ids) for ids, _ in data)
        ids = torch.tensor([[tokenizer.pad_token_id] * (length - len(x)) + x for x, _ in data])
        labels = torch.tensor([[IGNORE_INDEX] * (length - len(y)) + y for _, y in data])
        attention = torch.tensor([[0] * (length - len(x)) + [1] * len(x) for x, _ in data])
        return ids, attention, labels

    def loss_on(data):
        ids, attention, labels = collate(data)
        # Every answer fits in the last 32 tokens, so we only compute logits
        # there instead of a full-vocabulary tensor for every prompt token.
        assert (labels[:, :-32] == IGNORE_INDEX).all()
        logits = model(input_ids=ids, attention_mask=attention, logits_to_keep=33).logits[:, :-1, :]
        target = labels[:, -32:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=IGNORE_INDEX
        )

    def validation_loss():
        model.eval()
        with torch.no_grad():
            values = [float(loss_on(valid_set[i : i + 4])) for i in range(0, len(valid_set), 4)]
        return sum(values) / len(values)

    out.mkdir(parents=True)
    dump(out / "protocol.json", protocol)
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])

    start = time.monotonic()
    before = validation_loss()
    print(f"initial_validation_loss={before:.6f}", flush=True)
    random.Random(1729).shuffle(train_set)
    history = []
    model.train()
    for offset in range(0, len(train_set), cfg["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(train_set[offset : offset + cfg["batch_size"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step = {
            "step": offset // cfg["batch_size"] + 1,
            "loss": float(loss.detach()),
            "elapsed_seconds": time.monotonic() - start,
        }
        history.append(step)
        print(json.dumps(step), flush=True)

    after = validation_loss()
    model.save_pretrained(out, safe_serialization=True)
    changed = sum(
        int(not torch.equal(initial[n], p.detach())) for n, p in model.named_parameters() if p.requires_grad
    )
    write_rows(out / "training_history.jsonl", history)
    dump(
        out / "training_result.json",
        {
            "status": "COMPLETED",
            "actual_weight_updates": True,
            "changed_parameter_tensors": changed,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "steps": len(history),
            "train_examples": len(train_set),
            "validation_examples": len(valid_set),
            "validation_loss_before": before,
            "validation_loss_after": after,
            "elapsed_seconds": time.monotonic() - start,
            "paid_api_calls": 0,
            "training_reference": "weak observed-form labels, not indefinite gold",
            "protocol_sha256": digest((out / "protocol.json").read_bytes()),
            "model_manifest_sha256": digest((ROOT / "configs/cpu_model_manifest.json").read_bytes()),
            "adapter_sha256": digest((out / "adapter_model.safetensors").read_bytes()),
            "training_input_sha256": digest((LLM_DATA / "train_rows.jsonl").read_bytes()),
            "validation_input_sha256": digest((LLM_DATA / "dev_rows.jsonl").read_bytes()),
            "code_sha256": digest(pathlib.Path(__file__).read_bytes()),
        },
    )
    print("TRAINING COMPLETE", changed, "parameter tensors changed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CPU LLM inference and LoRA training.")
    parser.add_argument("command", choices=["infer", "train"])
    parser.add_argument("--model", required=True, help="local folder with base model weights")
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition", choices=["direct", "structured"], default="direct")
    parser.add_argument("--shots", choices=["zero", "corpus_few"], default="zero")
    parser.add_argument("--split", choices=["pilot", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adapter", help="LoRA adapter folder")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()

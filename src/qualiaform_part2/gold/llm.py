"""Run gold-set LLM conditions with a local Ollama model (or a mock).

Conditions are ``<task>_<shots>``:

==================  ==========================================================
annotate_zero/few   reproduce the qualia label (TELIC/AGENTIVE/HAVE/NONE)
direct_zero/few     predict BN vs. YI_CL_N directly
direct_reason_zero  direct prediction with a free-text reasoning step (ablation)
structured_zero/few annotate qualia first, then predict the form
==================  ==========================================================

Few-shot demonstrations are drawn **out of fold**: an item in Part 1 fold
*k* only ever sees gold examples from the other four folds, so no gold label
of the item (or of anything in its fold) reaches its prompt.

Decoding is greedy (temperature 0, fixed seed) and the output is constrained
by Ollama to a JSON schema with the label enums. Each answer is written to
disk immediately, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from ..extract import digest, dump, read_rows
from .codebook import FORMS, QUALIA, SCHEMAS, SYSTEM_PROMPTS, validate_output

CONDITIONS = (
    "annotate_zero",
    "annotate_few",
    "direct_zero",
    "direct_few",
    "direct_reason_zero",
    "structured_zero",
    "structured_few",
)
DEMOS_PER_PROMPT = 8
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def split_condition(condition: str) -> tuple[str, str]:
    task, shots = condition.rsplit("_", 1)
    if task not in SYSTEM_PROMPTS or shots not in ("zero", "few"):
        raise ValueError(f"unknown condition {condition!r}")
    return task, shots


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def model_input(item: dict[str, Any]) -> dict[str, str]:
    """The only item fields a model ever sees."""
    return {"verb": item["verb"], "noun": item["noun"], "context": item["masked_context"]}


def demo_output(task: str, item: dict[str, Any]) -> dict[str, Any]:
    """Gold answer for a demonstration, in the task's output format."""
    if task == "annotate":
        return {"reasoning": item["qualia_note"], "qualia": item["qualia_gold"]}
    if task == "direct":
        return {"form": item["form_target"]}
    if task == "structured":
        return {"reasoning": item["qualia_note"], "qualia": item["qualia_gold"], "form": item["form_target"]}
    raise ValueError(f"no demonstrations for task {task!r}")


def _rank(task: str, item: dict[str, Any]) -> str:
    """Deterministic pseudo-random order, independent of labels and file order."""
    return hashlib.sha256(f"demo-v1:{task}:{item['item_id']}".encode()).hexdigest()


def select_demos(task: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick a balanced set of :data:`DEMOS_PER_PROMPT` demonstrations from ``pool``.

    * annotate   – two per qualia label
    * direct     – four BN and four YI_CL_N
    * structured – one BN and one YI_CL_N per qualia label (topped up if a
      combination is missing)
    """
    ranked = sorted(pool, key=lambda item: _rank(task, item))
    if task == "annotate":
        cells = [(q, None) for q in QUALIA for _ in range(2)]
    elif task == "direct":
        cells = [(None, f) for f in FORMS for _ in range(4)]
    else:
        cells = [(q, f) for q in QUALIA for f in FORMS]

    chosen: list[dict[str, Any]] = []
    for qualia, form in cells:
        for item in ranked:
            if item in chosen:
                continue
            if (qualia is None or item["qualia_gold"] == qualia) and (
                form is None or item["form_target"] == form
            ):
                chosen.append(item)
                break
    for item in ranked:  # top up if some cell had no candidate
        if len(chosen) >= DEMOS_PER_PROMPT:
            break
        if item not in chosen:
            chosen.append(item)
    # Interleave labels rather than presenting them in blocks.
    return sorted(chosen, key=lambda item: _rank(task + ":order", item))


def build_messages(task: str, item: dict[str, Any], demos: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat = [{"role": "system", "content": SYSTEM_PROMPTS[task]}]
    for demo in demos:
        chat.append({"role": "user", "content": json.dumps(model_input(demo), ensure_ascii=False)})
        chat.append({"role": "assistant", "content": json.dumps(demo_output(task, demo), ensure_ascii=False)})
    chat.append({"role": "user", "content": json.dumps(model_input(item), ensure_ascii=False)})
    return chat


def build_requests(condition: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One request per item; few-shot demos come only from the other folds."""
    task, shots = split_condition(condition)
    requests = []
    for item in items:
        demos: list[dict[str, Any]] = []
        if shots == "few":
            pool = [other for other in items if other["fold"] != item["fold"]]
            demos = select_demos(task, pool)
        messages = build_messages(task, item, demos)
        requests.append(
            {
                "item_id": item["item_id"],
                "fold": item["fold"],
                "demo_ids": [d["item_id"] for d in demos],
                "messages": messages,
                "schema": SCHEMAS[task],
                "request_sha256": digest(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()),
            }
        )
    # Group by fold so the shared few-shot prefix can be reused by the server cache.
    return sorted(requests, key=lambda r: (r["fold"], r["item_id"]))


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


class OllamaClient:
    """Minimal client for a local Ollama server (standard library only)."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: float = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def describe(self) -> dict[str, Any]:
        """Model identity for the run manifest (name, digest, quantisation)."""
        try:
            tags = json.loads(urllib.request.urlopen(self.url + "/api/tags", timeout=10).read())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Install it from https://ollama.com, "
                f"start the app, then run: ollama pull {self.model}"
            ) from exc
        for entry in tags.get("models", []):
            if entry.get("name") in (self.model, self.model + ":latest") or entry.get("model") == self.model:
                details = entry.get("details", {})
                return {
                    "backend": "ollama",
                    "model": self.model,
                    "digest": entry.get("digest"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    "family": details.get("family"),
                }
        raise RuntimeError(f"Model {self.model!r} is not installed. Run: ollama pull {self.model}")

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": {"temperature": 0, "seed": 1729, "num_ctx": 8192, "num_predict": 256},
            },
        )
        return response["message"]["content"]


class MockClient:
    """Deterministic, label-blind answers for testing the pipeline only."""

    model = "mock"

    def describe(self) -> dict[str, Any]:
        return {"backend": "mock", "model": "mock", "digest": None}

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        seed = int(hashlib.sha256(messages[-1]["content"].encode()).hexdigest()[:8], 16)
        answer: dict[str, Any] = {}
        properties = schema["properties"]
        if "reasoning" in properties:
            answer["reasoning"] = "MOCK"
        if "qualia" in properties:
            answer["qualia"] = QUALIA[seed % 4]
        if "form" in properties:
            answer["form"] = FORMS[(seed // 4) % 2]
        return json.dumps(answer)


# --------------------------------------------------------------------------
# Running a condition
# --------------------------------------------------------------------------


def run_condition(
    condition: str,
    items: list[dict[str, Any]],
    client: Any,
    out_dir: pathlib.Path,
    progress: Callable[[str], None] = print,
) -> pathlib.Path:
    """Run (or resume) one condition; returns the run folder."""
    requests = build_requests(condition, items)
    identity = {
        "condition": condition,
        "model": client.describe(),
        "requests_sha256": digest(json.dumps([r["request_sha256"] for r in requests]).encode()),
        "count": len(requests),
    }
    folder = out_dir / condition
    folder.mkdir(parents=True, exist_ok=True)
    identity_file = folder / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError(f"{folder} was produced with a different model or prompts; use a new --out")
    else:
        dump(identity_file, identity)
        with (folder / "requests.jsonl").open("w", encoding="utf8") as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    predictions_file = folder / "predictions.jsonl"
    done = {r["item_id"] for r in read_rows(predictions_file)} if predictions_file.exists() else set()
    task, _ = split_condition(condition)
    start = time.monotonic()
    with predictions_file.open("a", encoding="utf8") as handle:
        for index, request in enumerate(requests, 1):
            if request["item_id"] in done:
                continue
            raw, parsed, error = "", None, None
            began = time.monotonic()
            try:
                raw = client.complete(request["messages"], request["schema"])
                parsed = validate_output(task, json.loads(raw))
            except (ValueError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "item_id": request["item_id"],
                "request_sha256": request["request_sha256"],
                "raw_response": raw,
                "prediction": parsed,
                "error": error,
                "seconds": round(time.monotonic() - began, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(requests):
                progress(f"  {condition}: {index}/{len(requests)}")

    predictions = read_rows(predictions_file)
    dump(
        folder / "manifest.json",
        {
            **identity,
            "completed": len(predictions) == len(requests),
            "invalid": sum(p["prediction"] is None for p in predictions),
            "seconds_this_invocation": round(time.monotonic() - start, 1),
            "paid_api_calls": 0,
        },
    )
    return folder

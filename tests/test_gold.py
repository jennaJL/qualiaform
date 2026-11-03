"""Tests for the gold-set LLM experiment (Part 3).

Synthetic items cover prompt construction, leakage and resumption. Tests on
the real gold set are skipped when the private data is not present.
"""

import csv
import json
import pathlib
import tempfile
import unittest

from qualiaform_part2.extract import read_rows
from qualiaform_part2.gold.codebook import CODEBOOK_EXAMPLES, SYSTEM_PROMPTS, validate_output
from qualiaform_part2.gold.llm import MockClient, build_requests, run_condition, select_demos

try:
    import numpy as np  # noqa: F401
    import sklearn  # noqa: F401

    from qualiaform_part2.gold.analyze import analyse, cohen_kappa, oof_logistic

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIVATE_ITEMS = ROOT / "data/private/gold_v1/items.jsonl"
PRIVATE_GOLD = ROOT / "data/private/part1/qualiaform_pilot.csv"
PRIVATE_FOLDS = ROOT / "data/private/part1/split_assignments.csv"
QUALIA = ["TELIC", "AGENTIVE", "HAVE", "NONE"]


def synthetic_items(n=40):
    items = []
    for k in range(n):
        form = "BN" if k % 4 == 0 else "YI_CL_N"
        items.append(
            {
                "item_id": f"T_{k:03d}",
                "fold": k % 5 + 1,
                "verb": "看",
                "noun": f"名词{k}",
                "masked_context": f"他看了[TARGET_NP]{k}。",
                "masked_span": f"一本名词{k}" if form == "YI_CL_N" else f"名词{k}",
                "form_target": form,
                "qualia_gold": QUALIA[k % 4],
                "typicality_gold": int(QUALIA[k % 4] in ("TELIC", "AGENTIVE")),
                "qualia_note": "示例。",
                "neg_dis": k % 2,
                "modifier": (k // 2) % 2,
                "aspect": (k // 3) % 2,
            }
        )
    return items


class PromptConstruction(unittest.TestCase):
    def test_few_shot_demos_come_from_other_folds_only(self):
        items = synthetic_items()
        folds = {i["item_id"]: i["fold"] for i in items}
        for condition in ("annotate_few", "direct_few", "structured_few"):
            for request in build_requests(condition, items):
                self.assertEqual(len(request["demo_ids"]), 8)
                self.assertNotIn(request["item_id"], request["demo_ids"])
                self.assertTrue(all(folds[d] != request["fold"] for d in request["demo_ids"]))

    def test_zero_shot_has_no_demos(self):
        for request in build_requests("structured_zero", synthetic_items()):
            self.assertEqual(request["demo_ids"], [])
            self.assertEqual(len(request["messages"]), 2)

    def test_item_labels_and_surface_form_never_reach_its_own_prompt(self):
        items = synthetic_items()
        by_id = {i["item_id"]: i for i in items}
        for condition in ("annotate_few", "direct_few", "structured_few", "direct_zero"):
            for request in build_requests(condition, items):
                item = by_id[request["item_id"]]
                final_user_message = json.loads(request["messages"][-1]["content"])
                self.assertEqual(set(final_user_message), {"verb", "noun", "context"})
                text = request["messages"][-1]["content"]
                self.assertNotIn(item["form_target"], text)
                self.assertNotIn(item["qualia_gold"], text)
                if item["form_target"] == "YI_CL_N":  # a BN span is just the noun, which is given
                    self.assertNotIn(item["masked_span"], text)

    def test_demo_selection_is_balanced(self):
        items = synthetic_items()
        annotate = select_demos("annotate", items)
        self.assertEqual(sorted(d["qualia_gold"] for d in annotate), sorted(QUALIA * 2))
        direct = select_demos("direct", items)
        self.assertEqual(sorted(d["form_target"] for d in direct), ["BN"] * 4 + ["YI_CL_N"] * 4)

    def test_requests_do_not_depend_on_gold_of_the_item(self):
        items = synthetic_items()
        before = [r["messages"][-1] for r in build_requests("direct_zero", items)]
        for item in items:
            item["form_target"] = "BN"
            item["qualia_gold"] = "NONE"
        after = [r["messages"][-1] for r in build_requests("direct_zero", items)]
        self.assertEqual(before, after)

    def test_output_validation(self):
        self.assertEqual(validate_output("direct", {"form": "BN"}), {"form": "BN"})
        with self.assertRaises(ValueError):
            validate_output("structured", {"qualia": "TELIC", "form": "bn"})
        with self.assertRaises(ValueError):
            validate_output("annotate", {"qualia": "PURPOSE"})

    def test_every_task_has_a_prompt(self):
        for prompt in SYSTEM_PROMPTS.values():
            self.assertIn("[TARGET_NP]", prompt)


class Running(unittest.TestCase):
    def test_resume_after_interruption_gives_identical_output(self):
        items = synthetic_items(20)
        with tempfile.TemporaryDirectory() as tmp:
            full = run_condition(
                "structured_few", items, MockClient(), pathlib.Path(tmp) / "a", progress=lambda m: None
            )
            partial_dir = pathlib.Path(tmp) / "b"
            run_condition("structured_few", items, MockClient(), partial_dir, progress=lambda m: None)
            # simulate an interruption: keep only the first 7 predictions
            lines = (partial_dir / "structured_few/predictions.jsonl").read_text().splitlines()[:7]
            (partial_dir / "structured_few/predictions.jsonl").write_text("\n".join(lines) + "\n")
            run_condition("structured_few", items, MockClient(), partial_dir, progress=lambda m: None)

            def answers(folder):
                return {r["item_id"]: r["prediction"] for r in read_rows(folder / "predictions.jsonl")}

            self.assertEqual(answers(full), answers(partial_dir / "structured_few"))
            self.assertTrue(json.loads((full / "manifest.json").read_text())["completed"])

    def test_resume_refuses_a_different_model(self):
        items = synthetic_items(10)

        class OtherMock(MockClient):
            def describe(self):
                return {"backend": "mock", "model": "other", "digest": None}

        with tempfile.TemporaryDirectory() as tmp:
            run_condition("direct_zero", items, MockClient(), pathlib.Path(tmp), progress=lambda m: None)
            with self.assertRaises(ValueError):
                run_condition("direct_zero", items, OtherMock(), pathlib.Path(tmp), progress=lambda m: None)


@unittest.skipUnless(HAVE_SKLEARN, "requires numpy and scikit-learn")
class Analysis(unittest.TestCase):
    def test_kappa_matches_scikit_learn(self):
        from sklearn.metrics import cohen_kappa_score

        gold = ["TELIC", "NONE", "NONE", "HAVE", "AGENTIVE", "NONE", "TELIC", "HAVE"]
        pred = ["TELIC", "NONE", "HAVE", "HAVE", "TELIC", "NONE", "NONE", "HAVE"]
        self.assertAlmostEqual(cohen_kappa(gold, pred, QUALIA), cohen_kappa_score(gold, pred))

    def test_analysis_runs_end_to_end_on_mock_predictions(self):
        items = synthetic_items()
        with tempfile.TemporaryDirectory() as tmp:
            runs = {}
            for condition in ("annotate_few", "direct_few", "structured_few"):
                folder = run_condition(
                    condition, items, MockClient(), pathlib.Path(tmp), progress=lambda m: None
                )
                runs[condition] = {
                    r["item_id"]: r["prediction"] for r in read_rows(folder / "predictions.jsonl")
                }
            summary, systems = analyse(items, runs)
        self.assertIn("LLM structured_few", summary["form_prediction"])
        self.assertIn("logreg: context + LLM qualia (annotate_few)", systems)
        self.assertEqual(summary["n_items"], 40)


@unittest.skipUnless(PRIVATE_ITEMS.exists() and HAVE_SKLEARN, "private gold data not present")
class RealGoldSet(unittest.TestCase):
    def test_codebook_examples_are_not_gold_items(self):
        items = read_rows(PRIVATE_ITEMS)
        pairs = {(i["verb"], i["noun"]) for i in items}
        phrases = [i["mandarin_vn"] for i in items]
        for examples in CODEBOOK_EXAMPLES.values():
            for verb, noun in examples:
                self.assertNotIn((verb, noun), pairs)
                self.assertFalse(any(verb + noun in p for p in phrases), verb + noun)

    def test_masks_hide_the_numeral_for_every_yi_item(self):
        for item in read_rows(PRIVATE_ITEMS):
            self.assertEqual(item["masked_context"].count("[TARGET_NP]"), 1)
            if item["form_target"] == "YI_CL_N" and item["mask_adjustment"] != "no_yi_found":
                self.assertTrue(item["masked_span"].startswith("一"), item["item_id"])

    def test_logistic_protocol_reproduces_part1_scores(self):
        """Same folds and model as Part 1 → same out-of-fold balanced accuracy on all 156 items."""
        with open(PRIVATE_FOLDS, encoding="utf-8-sig") as handle:
            folds = {
                r["item_id"]: int(r["fold"])
                for r in csv.DictReader(handle)
                if r["protocol"] == "grouped" and r["seed"] == "42"
            }
        with open(PRIVATE_GOLD, encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        items = [
            {
                "fold": folds[r["item_id"]],
                "form_target": r["form_target"],
                **{f: int(r[f]) for f in ("neg_dis", "modifier", "aspect")},
            }
            for r in rows
        ]
        gold = [r["form_target"] for r in rows]

        def balanced_accuracy(pred):
            return (
                sum(
                    sum(p == g == label for p, g in zip(pred, gold)) / gold.count(label)
                    for label in ("BN", "YI_CL_N")
                )
                / 2
            )

        self.assertAlmostEqual(balanced_accuracy(oof_logistic(items, None)), 0.6512096774193548)
        qualia = [r["qualia_gold"] for r in rows]
        self.assertAlmostEqual(balanced_accuracy(oof_logistic(items, qualia)), 0.7147177419354839)


class OllamaClientContract(unittest.TestCase):
    """Checks the HTTP payload against a local stand-in for the Ollama server."""

    def test_chat_request_payload_and_model_lookup(self):
        import http.server
        import threading

        from qualiaform_part2.gold.codebook import SCHEMAS
        from qualiaform_part2.gold.llm import OllamaClient

        received = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._reply({"models": [{"name": "qwen2.5:7b-instruct", "digest": "abc", "details": {}}]})

            def do_POST(self):
                received.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self._reply({"message": {"role": "assistant", "content": '{"form": "BN"}'}})

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OllamaClient(url=f"http://127.0.0.1:{server.server_port}")
            self.assertEqual(client.describe()["digest"], "abc")
            raw = client.complete([{"role": "user", "content": "x"}], SCHEMAS["direct"])
        finally:
            server.shutdown()
        self.assertEqual(json.loads(raw), {"form": "BN"})
        self.assertEqual(received["model"], "qwen2.5:7b-instruct")
        self.assertEqual(received["format"], SCHEMAS["direct"])
        self.assertFalse(received["stream"])
        self.assertEqual(received["options"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()

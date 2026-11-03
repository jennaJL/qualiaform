"""Real-model in-process integration tests and descriptive local latency."""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from fastapi.testclient import TestClient

from qualiaform_part2.experiment import model_input
from qualiaform_part2.extract import dump, read_rows
from qualiaform_part2.service import ModelService, create_app

root = pathlib.Path(__file__).resolve().parents[1]
service = ModelService()
rows = read_rows(root / "data/processed/ud_candidates_v1/candidates.jsonl")
test = [r for r in rows if r["source_split"] == "test"]
expected = read_rows(service.benchmark / service.name / "test_predictions.jsonl")
check = []
latencies = []
with TestClient(create_app(service)) as client:
    assert client.get("/healthz").status_code == 200
    for r, p in zip(test[:20], expected[:20]):
        started = time.perf_counter()
        res = client.post("/predict", json=model_input(r))
        latencies.append((time.perf_counter() - started) * 1000)
        assert res.status_code == 200, res.text
        got = res.json()
        assert got["label"] == p["prediction"]
        assert abs(got["yi_score_uncalibrated"] - p["yi_score_uncalibrated"]) < 1e-5
        assert all(n["source_split"] == "train" for n in got["nearest_training_examples"])
        check.append(r["candidate_id"])
    batch = client.post("/predict_batch", json={"items": [model_input(r) for r in test[:4]]})
    assert batch.status_code == 200
    assert [r["label"] for r in batch.json()] == [r["prediction"] for r in expected[:4]]
    assert client.post("/predict", json={**model_input(test[0]), "observed_form": "BN"}).status_code == 422
    (root / "reports/api_openapi.json").write_text(
        json.dumps(client.get("/openapi.json").json(), ensure_ascii=False, indent=2) + "\n"
    )
    metrics = client.get("/metrics").text
    (root / "reports/api_metrics_example.txt").write_text(metrics)
    (root / "reports/api_response_example.json").write_text(
        json.dumps(client.post("/predict", json=model_input(test[0])).json(), ensure_ascii=False, indent=2)
        + "\n"
    )
result = {
    "status": "PASS",
    "model": service.name,
    "real_model_examples_checked": 20,
    "batch_checked": 4,
    "prediction_matches_offline_benchmark": True,
    "nearest_examples_train_only": True,
    "in_process_TestClient": True,
    "latency_ms": {
        "n": len(latencies),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    },
    "latency_scope": "sequential warm/local CPU prototype, 20 requests including first inference, no load or production SLO claim",
}
dump(root / "reports/service_verification.json", result)
print(json.dumps(result, indent=2))

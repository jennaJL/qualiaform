"""Local FastAPI inference service for the selected PyTorch classifier.

Endpoints:

* ``GET  /``              – small web page for manual testing
* ``GET  /healthz``       – readiness check
* ``POST /predict``       – one item
* ``POST /predict_batch`` – up to 16 items
* ``GET  /benchmark``     – stored test metrics and bootstrap intervals
* ``GET  /metrics``       – Prometheus-style request counters

Every prediction also returns the three most similar *training* examples
(cosine similarity of encoder features). This is nearest-neighbour lookup
for inspection, not retrieval-augmented generation.

User text is never written to logs. Start with ``python scripts/serve.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .extract import LABELS, read_rows
from .portfolio import ROOT, Encoder, head, input_text, sha_file

LOG = logging.getLogger("qualiaform.requests")
NOTICE = (
    "Predicts extracted observed form, not indefinite eligibility. "
    "Scores are uncalibrated; expert review is required."
)
NEIGHBOUR_FIELDS = [
    "candidate_id",
    "sentence",
    "masked_context",
    "noun_lemma",
    "verb_lemma",
    "relation",
    "observed_form",
]


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------


class PredictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    masked_context: Annotated[str, Field(min_length=1, max_length=1200)]
    noun_lemma: Annotated[str, Field(min_length=1, max_length=80)]
    verb_lemma: Annotated[str, Field(min_length=1, max_length=80)]
    relation: Annotated[str, Field(pattern=r"^(obj|iobj|nsubj|obl)(:[a-z]+)?$")] = "obj"

    @field_validator("masked_context")
    @classmethod
    def single_target(cls, value: str) -> str:
        if value.count("[TARGET_NP]") != 1:
            raise ValueError("Use exactly one [TARGET_NP] marker")
        return value


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: Annotated[list[PredictInput], Field(min_length=1, max_length=16)]


class Prediction(BaseModel):
    label: Literal["BN", "YI_CL_N"]
    yi_score_uncalibrated: float
    decision_threshold: float
    model_id: str
    input_truncated: bool
    nearest_training_examples: list[dict]
    notice: str = NOTICE


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------


class ModelService:
    """Loads the selected head, the frozen encoder and train-split features.

    Every artefact is checksum-verified on start-up.
    """

    def __init__(self, benchmark: str | None = None, encoder: str | None = None):
        import torch
        from safetensors.torch import load_file

        self.torch = torch
        self.benchmark = pathlib.Path(
            benchmark or os.environ.get("QUALIAFORM_BENCHMARK", ROOT / "results/portfolio_v1")
        )
        self.selection = json.loads((self.benchmark / "selection.json").read_text())
        self.name = self.selection["selected"]
        folder = self.benchmark / self.name
        self.meta = json.loads((folder / "model.json").read_text())
        if sha_file(folder / "model.safetensors") != self.meta["weights_sha256"]:
            raise ValueError("Classifier checksum mismatch")

        self.encoder = Encoder(encoder)
        self._verify_encoder()

        state = load_file(str(folder / "model.safetensors"))
        self.mean = state.pop("feature_mean")
        self.std = state.pop("feature_std")
        self.model = head(
            self.meta["kind"], self.meta["input_dim"], self.meta["hidden"], self.meta["dropout"]
        )
        self.model.load_state_dict(state)
        self.model.eval()

        self.train_features, self.train_rows = self._load_training_examples()
        self.lock = threading.Lock()

    def _verify_encoder(self) -> None:
        manifest = json.loads((ROOT / "configs/encoder_manifest.json").read_text())
        for name, info in manifest["files"].items():
            if sha_file(self.encoder.path / name) != info["sha256"]:
                raise ValueError("Encoder checksum mismatch: " + name)

    @staticmethod
    def _load_training_examples() -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Cached encoder features and display fields for the train split only."""
        feature_dir = ROOT / "data/portfolio_features_v1"
        manifest = json.loads((feature_dir / "manifest.json").read_text())
        if sha_file(feature_dir / "features.npz") != manifest["features_sha256"]:
            raise ValueError("Feature cache checksum mismatch")
        data = np.load(feature_dir / "features.npz", allow_pickle=False)
        rows = read_rows(ROOT / "data/processed/ud_candidates_v1/candidates.jsonl")
        if list(data["ids"]) != [r["candidate_id"] for r in rows]:
            raise ValueError("Feature/row ID mismatch")
        train_ix = [i for i, r in enumerate(rows) if r["source_split"] == "train"]
        examples = [{k: rows[i][k] for k in NEIGHBOUR_FIELDS} for i in train_ix]
        return data["x"][train_ix], examples

    def predict(self, inputs: list[Any]) -> list[dict[str, Any]]:
        rows = [item.model_dump() if hasattr(item, "model_dump") else item for item in inputs]
        with self.lock:
            x = self.encoder.encode(rows)
            truncated = [
                len(self.encoder.tokenizer.encode(input_text(r))) > self.encoder.max_length for r in rows
            ]
            with self.torch.inference_mode():
                z = (self.torch.from_numpy(x) - self.mean) / self.std
                scores = self.model(z).softmax(-1)[:, 1].numpy()

        results = []
        for vector, yi_score, was_truncated in zip(x, scores, truncated):
            similarities = self.train_features @ vector  # features are L2-normalised
            top = np.argsort(-similarities)[:3]
            neighbours = [
                {**self.train_rows[i], "cosine_similarity": float(similarities[i]), "source_split": "train"}
                for i in top
            ]
            results.append(
                {
                    "label": LABELS[int(yi_score >= self.meta["threshold"])],
                    "yi_score_uncalibrated": float(yi_score),
                    "decision_threshold": self.meta["threshold"],
                    "model_id": self.name,
                    "input_truncated": was_truncated,
                    "nearest_training_examples": neighbours,
                    "notice": NOTICE,
                }
            )
        return results

    def benchmark_summary(self) -> dict[str, Any]:
        return {
            "selected": self.name,
            "selection_basis": "development macro-F1, seed 1729",
            "test_n": 913,
            "reference": "weak observed form",
            "test_reused": True,
            "models": json.loads((self.benchmark / "test_metrics.json").read_text()),
            "bootstrap": json.loads((self.benchmark / "selected_bootstrap.json").read_text()),
        }


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(service: Any = None) -> FastAPI:
    """Build the FastAPI app; ``service`` can be a test double."""
    stats = {"requests": 0, "errors": 0, "seconds": 0.0}
    lock = threading.Lock()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = service or ModelService()
        yield

    app = FastAPI(title="QualiaForm Research API", version="2.0.0", description=NOTICE, lifespan=lifespan)

    @app.middleware("http")
    async def telemetry(request: Request, call_next):
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            with lock:
                stats["requests"] += 1
                stats["errors"] += int(status >= 400)
                stats["seconds"] += elapsed
            # Log request metadata only – never user text or model inputs.
            LOG.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "seconds": round(elapsed, 6),
                    }
                )
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page():
        return (pathlib.Path(__file__).parent / "static/index.html").read_text(encoding="utf8")

    @app.get("/healthz")
    def health():
        return {
            "status": "ready",
            "model": app.state.model.name,
            "backend": "local_cpu",
            "reference": "weak_observed_form",
        }

    @app.post("/predict", response_model=Prediction)
    def predict(item: PredictInput):
        return app.state.model.predict([item])[0]

    @app.post("/predict_batch", response_model=list[Prediction])
    def predict_batch(batch: BatchInput):
        return app.state.model.predict(batch.items)

    @app.get("/benchmark")
    def benchmark():
        return app.state.model.benchmark_summary()

    @app.get("/metrics", response_class=PlainTextResponse)
    def monitoring():
        with lock:
            snapshot = dict(stats)
        return (
            "# TYPE qualiaform_requests_total counter\n"
            f"qualiaform_requests_total {snapshot['requests']}\n"
            "# TYPE qualiaform_errors_total counter\n"
            f"qualiaform_errors_total {snapshot['errors']}\n"
            "# TYPE qualiaform_request_seconds_sum counter\n"
            f"qualiaform_request_seconds_sum {snapshot['seconds']:.6f}\n"
        )

    return app

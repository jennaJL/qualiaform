"""Reproduce four real prompt conditions plus a small LoRA training/evaluation run."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
a = p.parse_args()
model = str(pathlib.Path(a.model).resolve())
os.chdir(root)
os.environ.update(
    PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
)
folder = root / "models/lora_cpu_v2"
summary = folder / "training_result.json"
if summary.exists():
    result = json.loads(summary.read_text())
    assert result["status"] == "COMPLETED" and result["actual_weight_updates"]
    assert (
        hashlib.sha256((folder / "adapter_model.safetensors").read_bytes()).hexdigest()
        == result["adapter_sha256"]
    )
else:
    subprocess.run(
        [sys.executable, "-m", "qualiaform_part2.local_cpu", "train", "--model", model, "--out", str(folder)],
        check=True,
    )
subprocess.run(
    [sys.executable, str(root / "scripts/run_local_suite.py"), "--model", model, "--split", "test"],
    check=True,
)
out = "results/local_v2_test_lora_direct"
subprocess.run(
    [
        sys.executable,
        "-m",
        "qualiaform_part2.local_cpu",
        "infer",
        "--model",
        model,
        "--out",
        out,
        "--adapter",
        str(folder),
    ],
    check=True,
)
subprocess.run([sys.executable, "-m", "qualiaform_part2.evaluate", "--run", out], check=True)
print("COMPLETE: four inference conditions + actual LoRA + held-out evaluation", flush=True)

"""Runs all four conditions without a model. Outputs never overwrite existing runs."""

import argparse
import os
import pathlib
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="results/mock")
args = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ["PYTHONPATH"] = str(root / "src")
for condition in ("direct", "structured"):
    for shots in ("zero", "few"):
        name = f"{condition}_{shots}"
        out = f"{args.prefix}_{name}"
        cmd = [
            sys.executable,
            "-m",
            "qualiaform_part2.experiment",
            "--config",
            f"configs/{name}.json",
            "--mode",
            "mock",
            "--out",
            out,
        ]
        if shots == "few":
            cmd += ["--demo-file", "tests/synthetic_demos.jsonl"]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, "-m", "qualiaform_part2.evaluate", "--run", out], check=True)

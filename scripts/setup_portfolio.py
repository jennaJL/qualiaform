"""Install free runtime and download checksum-pinned encoder. No API key."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import venv

root = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--env-dir", default=str(root / ".venv-portfolio"))
p.add_argument("--model-dir", default=str(root / "models/minilm_base"))
p.add_argument("--skip-install", action="store_true")
a = p.parse_args()
if not a.skip_install:
    env = pathlib.Path(a.env_dir)
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    subprocess.run(
        [
            str(env / "bin/python"),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(root / "requirements-portfolio.txt"),
        ],
        check=True,
    )
m = json.loads((root / "configs/encoder_manifest.json").read_text())
dest = pathlib.Path(a.model_dir)
dest.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


for name, info in m["files"].items():
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if sha(out) != info["sha256"]:
            raise ValueError("Existing encoder file hash mismatch: " + name)
        continue
    temp = pathlib.Path(str(out) + ".part")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "3",
            f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{name}",
            "-o",
            str(temp),
        ],
        check=True,
    )
    if sha(temp) != info["sha256"]:
        raise ValueError("Encoder download hash mismatch: " + name)
    temp.rename(out)
print("Ready. Encoder:", dest)

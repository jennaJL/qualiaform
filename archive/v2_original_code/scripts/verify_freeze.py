"""Check frozen source/outputs and byte-for-byte extraction reproduction in a temporary dir."""
import json, os, pathlib, subprocess, sys, tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from qualiaform_part2.extract import digest
root=pathlib.Path(__file__).resolve().parents[1];os.chdir(root)
frozen=root/'data/processed/ud_candidates_v1';manifest=json.loads((frozen/'manifest.json').read_text())
for name,sha in manifest['input_sha256'].items():
    assert digest((root/'data/ud'/name).read_bytes())==sha,name
for name,sha in manifest['output_sha256'].items():
    assert digest((frozen/name).read_bytes())==sha,name
assert digest((root/'src/qualiaform_part2/extract.py').read_bytes())==manifest['code_sha256']
with tempfile.TemporaryDirectory() as tmp:
    dest=pathlib.Path(tmp)/'reproduced'
    env=dict(os.environ,PYTHONPATH=str(root/'src'))
    subprocess.run([sys.executable,'-m','qualiaform_part2.extract','--out',str(dest)],env=env,check=True,capture_output=True)
    for name in ('candidates.jsonl','candidates.csv','statistics.json'):
        assert (dest/name).read_bytes()==(frozen/name).read_bytes(),name
print('PASS: all source and output hashes verified; independent extraction byte-identical.')

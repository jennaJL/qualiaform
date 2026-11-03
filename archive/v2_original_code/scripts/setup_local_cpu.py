"""Download pinned public weights with checksum verification; no account/API key."""
import argparse,hashlib,json,pathlib,subprocess,sys,venv
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model-dir',default=str(root/'models/base_qwen05'));p.add_argument('--env-dir',default=str(root/'.venv-local'));p.add_argument('--skip-install',action='store_true');a=p.parse_args()
if not a.skip_install:
    env=pathlib.Path(a.env_dir)
    if not env.exists():venv.EnvBuilder(with_pip=True).create(env)
    subprocess.run([str(env/'bin/python'),'-m','pip','install','--no-cache-dir','-r',str(root/'requirements-local-cpu-lock.txt')],check=True)
manifest=json.loads((root/'configs/cpu_model_manifest.json').read_text());dest=pathlib.Path(a.model_dir);dest.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for name,info in manifest['files'].items():
    output=dest/name
    if output.exists():
        if sha(output)!=info['sha256']:raise ValueError('Existing model file checksum mismatch: '+name)
        continue
    partial=pathlib.Path(str(output)+'.part')
    subprocess.run(['curl','-fL','--retry','3',f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}",'-o',str(partial)],check=True)
    if sha(partial)!=info['sha256']:raise ValueError('Download checksum mismatch: '+name)
    partial.rename(output)
print('Model verified at',dest)

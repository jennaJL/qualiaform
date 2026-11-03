"""One command: frozen HF features -> PyTorch benchmark. Existing completed runs are verified, not overwritten."""
import argparse,os,pathlib,subprocess,sys,json,hashlib
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--features',default='data/portfolio_features_v1');p.add_argument('--out',default='results/portfolio_v1');p.add_argument('--encoder');a=p.parse_args();os.chdir(root)
os.environ.update(PYTHONPATH=str(root/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def run(cmd):subprocess.run([sys.executable,'-m','qualiaform_part2.portfolio']+cmd,check=True)
if not (pathlib.Path(a.features)/'manifest.json').exists():
    cmd=['features','--out',a.features]
    if a.encoder:cmd+=['--encoder',str(pathlib.Path(a.encoder).resolve())]
    run(cmd)
if not (pathlib.Path(a.out)/'manifest.json').exists():run(['train','--features',a.features,'--out',a.out])
else:
    m=json.loads((pathlib.Path(a.out)/'manifest.json').read_text())
    assert m['status']=='COMPLETED'
    assert m['dataset_sha256']==hashlib.sha256((root/'data/processed/ud_candidates_v1/candidates.jsonl').read_bytes()).hexdigest()
    print('Existing completed benchmark verified:',a.out)

"""Run four real CPU conditions sequentially; interruption-safe per-batch outputs."""
import argparse,pathlib,os,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--split',choices=['pilot','test'],default='test');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--prefix',default='local_v2');a=p.parse_args()
model=str(pathlib.Path(a.model).resolve());os.chdir(root)
os.environ.update(PYTHONPATH=str(root/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for condition in ('direct','structured'):
 for shots in ('zero','corpus_few'):
  out=f'results/{a.prefix}_{a.split}_{condition}_{shots}'
  subprocess.run([sys.executable,'-m','qualiaform_part2.local_cpu','infer','--model',model,'--out',out,'--split',a.split,'--condition',condition,'--shots',shots,'--batch-size',str(a.batch_size)],check=True)
  subprocess.run([sys.executable,'-m','qualiaform_part2.evaluate','--run',out],check=True)

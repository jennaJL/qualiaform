"""Deterministic workload subset, explicit smoke demos, experiment configurations."""
import collections, csv, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from qualiaform_part2.extract import read_rows,write_rows,dump,digest
root=pathlib.Path(__file__).resolve().parents[1]
for condition in ('direct','structured'):
    for shots in ('zero','few'):
        dump(root/f'configs/{condition}_{shots}.json',{'condition':condition,'shots':shots,'split':'test',
            'prompt':f'prompts/{condition}_v1.txt','generation':{'temperature':0,'seed':1729,'max_output_tokens':300},'timeout_seconds':60})
rows=read_rows(root/'data/processed/ud_candidates_v1/candidates.jsonl')
subset=sorted(sorted(rows,key=lambda r:digest(('sample-v1:'+r['candidate_id']).encode()))[:1200],key=lambda r:r['candidate_id'])
p=root/'data/processed/working_1200_v1'
if not p.exists():
    p.mkdir();write_rows(p/'candidates.jsonl',subset)
    with (p/'candidates.csv').open('w',newline='',encoding='utf8') as f:
        fields=['candidate_id','source_split','sentence','verb','noun','relation','observed_form','classifier','masked_context','indefinite_status']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in subset)
    dump(p/'manifest.json',{'n':1200,'selection':'smallest SHA256(sample-v1:+candidate_id); no class rebalancing; not a new split',
        'parent_sha256':digest((root/'data/processed/ud_candidates_v1/candidates.jsonl').read_bytes()),
        'sha256':digest((p/'candidates.jsonl').read_bytes()),'counts':dict(collections.Counter(r['source_split']+'/'+r['observed_form'] for r in subset))})
examples=[{'masked_context':'小明去商店买[TARGET_NP]。','noun_lemma':'书','verb_lemma':'买','relation':'obj'},
          {'masked_context':'她每天练习弹[TARGET_NP]。','noun_lemma':'钢琴','verb_lemma':'弹','relation':'obj'}]
write_rows(root/'tests/synthetic_demos.jsonl',[{'candidate_id':f'synthetic-{i}','provenance':'synthetic_smoke','input':inp,
    'output':{'form_prediction':['YI_CL_N','BN'][i],'typicality':'typical','qualia':['NONE','TELIC'][i],'evidence':'SYNTHETIC schema fixture; not gold'}} for i,inp in enumerate(examples)])
print('Created configurations, synthetic fixtures and frozen 1200-item subset.')

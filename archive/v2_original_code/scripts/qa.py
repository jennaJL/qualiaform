"""Read-only semantic/data-flow checks over actual artifacts."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from qualiaform_part2.extract import read_rows,dump,digest
root=pathlib.Path(__file__).resolve().parents[1]
rows=read_rows(root/'data/processed/ud_candidates_v1/candidates.jsonl');by={r['candidate_id']:r for r in rows}
checks={'unique_candidate_ids':len(by)==len(rows),'human_gold_not_fabricated':all(r['indefinite_status']=='unreviewed' and r['gold_typicality'] is None and r['gold_qualia'] is None for r in rows),'single_mask_every_row':all(r['masked_context'].count('[TARGET_NP]')==1 for r in rows)}
for run in sorted((root/'results').glob('mock_v2_*')):
    req=read_rows(run/'requests.jsonl');pred=read_rows(run/'predictions.jsonl');m=json.loads((run/'manifest.json').read_text())
    checks[run.name+'/913_complete']=len(req)==len(pred)==913
    checks[run.name+'/no_raw_target_fields']=all(set(r['request']['input'])=={'masked_context','noun_lemma','verb_lemma','relation'} for r in req)
    checks[run.name+'/request_hashes']=all(digest(json.dumps(r['request'],ensure_ascii=False,sort_keys=True).encode())==r['request_sha256'] for r in req)
    checks[run.name+'/no_invalid']=all(p['error'] is None for p in pred)
    checks[run.name+'/truthful_mock_status']=m['execution_mode']=='mock' and not m['scientific_model_experiment_run']
assert all(checks.values()),checks
dump(root/'reports/quality_checks.json',{'status':'PASS','checks':checks,'scope':'software/data invariants, NOT linguistic-gold validation'})
print(f'PASS: {len(checks)} dataset/run QA checks')

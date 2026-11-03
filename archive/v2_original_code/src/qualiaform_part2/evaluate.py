"""Strict ID join, invalid responses count as errors; no fabricated semantic gold."""
import argparse, collections, csv, json, pathlib
from .extract import LABELS, read_rows, dump, digest

def metrics(pairs):
    matrix={y:{p:0 for p in (*LABELS,'INVALID')} for y in LABELS}
    for y,p in pairs:
        if y not in LABELS: raise ValueError('unknown reference label')
        matrix[y][p if p in LABELS else 'INVALID']+=1
    n=len(pairs); per={}
    for label in LABELS:
        tp=matrix[label][label]; support=sum(matrix[label].values()); predicted=sum(matrix[y][label] for y in LABELS)
        precision=tp/predicted if predicted else 0.0; recall=tp/support if support else None
        f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.0
        per[label]={'precision':precision,'recall':recall,'f1':f1,'support':support}
    recalls=[v['recall'] for v in per.values() if v['recall'] is not None]
    return {'n':n,'accuracy':sum(matrix[l][l] for l in LABELS)/n if n else None,
            'balanced_accuracy':sum(recalls)/len(recalls) if len(recalls)==2 else None,
            'macro_f1':sum(v['f1'] for v in per.values())/2 if n else None,'per_class':per,
            'confusion_matrix':matrix,'invalid':sum(matrix[l]['INVALID'] for l in LABELS),
            'metric_policy':'zero precision/F1 for undefined denominator; balanced accuracy null if either reference class absent'}

def evaluate(rows,predictions):
    by={r['candidate_id']:r for r in rows}; ids=[p['candidate_id'] for p in predictions]
    if len(by)!=len(rows) or len(set(ids))!=len(ids): raise ValueError('duplicate ID')
    if set(ids)-set(by): raise ValueError('unknown prediction ID')
    pairs=[]; errors=[]; groups=collections.defaultdict(list)
    for p in predictions:
        r=by[p['candidate_id']]; obj=p.get('prediction') or {}; pred=obj.get('form_prediction','INVALID'); y=r['observed_form']
        pair=(y,pred); pairs.append(pair)
        for field in ('source_split','relation','noun_lemma'):
            groups[field+':'+r[field]].append(pair)
        for field in ('typicality','qualia'):
            groups['PREDICTED_'+field+':'+str(obj.get(field,'unavailable'))].append(pair)
        if y!=pred:
            errors.append({'candidate_id':r['candidate_id'],'sentence':r['sentence'],'reference_observed_form':y,'prediction':pred,
                           'predicted_typicality':obj.get('typicality'),'predicted_qualia':obj.get('qualia'),
                           'flags':'|'.join(r['flags']),'error_type':'invalid_response' if pred not in LABELS else 'unreviewed',
                           'reviewer':'','review_notes':'','indefinite_gold':'unavailable'})
    return {'overall':metrics(pairs),'slices':{k:metrics(v) for k,v in sorted(groups.items())}},errors

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='data/processed/ud_candidates_v1/candidates.jsonl'); p.add_argument('--run',required=True); a=p.parse_args()
    run=pathlib.Path(a.run); manifest=json.loads((run/'manifest.json').read_text())
    if digest(pathlib.Path(a.dataset).read_bytes())!=manifest['dataset_sha256']: raise ValueError('dataset hash mismatch')
    predictions=read_rows(run/'predictions.jsonl'); requests=read_rows(run/'requests.jsonl')
    if {x['candidate_id'] for x in predictions}!={x['candidate_id'] for x in requests}: raise ValueError('missing/extra predictions')
    if len(predictions)!=manifest['count'] or len(requests)!=manifest['count']: raise ValueError('run count mismatch')
    request_hash={x['candidate_id']:x['request_sha256'] for x in requests}
    if any(x['request_sha256']!=request_hash[x['candidate_id']] for x in predictions): raise ValueError('request mismatch')
    results,errors=evaluate(read_rows(a.dataset),predictions)
    results.update(execution_mode=manifest['execution_mode'], interpretation='SMOKE ONLY; not LLM performance' if manifest['execution_mode']=='mock' else 'Agreement with extracted observed form; not indefinite-gold accuracy')
    dump(run/'metrics.json',results)
    fields=['candidate_id','sentence','reference_observed_form','prediction','predicted_typicality','predicted_qualia','flags','error_type','reviewer','review_notes','indefinite_gold']
    with (run/'errors.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(errors)
    print(json.dumps(results['overall'],indent=2))
if __name__=='__main__': main()

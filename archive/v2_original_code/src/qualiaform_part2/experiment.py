"""Offline reproducible requests and predictions; optional external model command."""
import argparse, collections, hashlib, json, pathlib, subprocess
from .extract import read_rows, write_rows, dump, digest, LABELS
SAFE_FIELDS=('masked_context','noun_lemma','verb_lemma','relation')
def model_input(row):
    return {k:row[k] for k in SAFE_FIELDS}
def validate_response(obj, condition):
    if not isinstance(obj,dict) or obj.get('form_prediction') not in LABELS: raise ValueError('invalid form_prediction')
    if condition=='structured':
        if obj.get('typicality') not in ('typical','atypical','uncertain'): raise ValueError('invalid typicality')
        if obj.get('qualia') not in ('TELIC','AGENTIVE','HAVE','NONE'): raise ValueError('invalid qualia')
        if not isinstance(obj.get('evidence'),str): raise ValueError('missing evidence')
    return obj

def demonstrations(path, targets, condition, allow_synthetic=False):
    demos=read_rows(path)
    if not demos: raise ValueError('few-shot requires nonempty demonstrations')
    target_contexts={r['masked_context'] for r in targets}
    target_hashes={r['sentence_hash'] for r in targets}
    target_ids={r['candidate_id'] for r in targets}
    for d in demos:
        if d.get('provenance') not in ('part1_gold','synthetic_smoke'): raise ValueError('demonstration provenance required')
        if d['provenance']=='synthetic_smoke' and not allow_synthetic: raise ValueError('synthetic examples allowed only in mock smoke runs')
        if d.get('candidate_id') in target_ids or d.get('sentence_hash') in target_hashes or d['input']['masked_context'] in target_contexts:
            raise ValueError('demonstration overlaps evaluation target')
        if set(d['input'])!=set(SAFE_FIELDS): raise ValueError('unsafe demonstration fields')
        if d['provenance']=='part1_gold' and not d.get('sentence_hash'): raise ValueError('gold demo needs original sentence hash for overlap checks')
        validate_response(d['output'],condition)
    return [{'input':d['input'],'output':d['output'] if condition=='structured' else {'form_prediction':d['output']['form_prediction']}} for d in demos]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--dataset',default='data/processed/ud_candidates_v1/candidates.jsonl')
    p.add_argument('--out',required=True); p.add_argument('--mode',choices=['prepare','mock','external'],default='prepare')
    p.add_argument('--demo-file'); p.add_argument('--limit',type=int); p.add_argument('--model-id'); p.add_argument('--adapter',nargs='+')
    a=p.parse_args(); cfg=json.loads(pathlib.Path(a.config).read_text()); out=pathlib.Path(a.out)
    if out.exists(): raise SystemExit('Refusing to overwrite experiment')
    if a.limit is not None and a.limit<=0: raise ValueError('limit must be positive')
    if a.mode=='external' and (not a.adapter or not a.model_id): raise ValueError('external requires --adapter and --model-id')
    all_rows=read_rows(a.dataset)
    # Deduplicate normalized sentence strings across source splits, prioritizing train, then dev.
    train_hash={r['sentence_hash'] for r in all_rows if r['source_split']=='train'}
    dev_hash={r['sentence_hash'] for r in all_rows if r['source_split']=='dev'}
    rows=[r for r in all_rows if r['source_split']==cfg['split'] and r['sentence_hash'] not in (train_hash|dev_hash if cfg['split']=='test' else train_hash if cfg['split']=='dev' else set())]
    seen=set(); unique=[]
    for r in rows:
        key=(r['sentence_hash'],r['noun_token_id'])
        if key not in seen: unique.append(r); seen.add(key)
    rows=unique[:a.limit] if a.limit else unique
    if not rows: raise ValueError('empty evaluation set')
    prompt=pathlib.Path(cfg['prompt']).read_text(encoding='utf8')
    demos=[]
    if cfg['shots']=='few':
        if not a.demo_file: raise ValueError('few-shot requires --demo-file; Part 1 gold unavailable')
        demos=demonstrations(a.demo_file,all_rows,cfg['condition'],allow_synthetic=a.mode=='mock')
    out.mkdir(parents=True)
    requests=[]; predictions=[]
    for r in rows:
        request={'system':prompt,'demonstrations':demos,'input':model_input(r),'generation':cfg['generation']}
        encoded=json.dumps(request,ensure_ascii=False,sort_keys=True)
        req_hash=digest(encoded.encode())
        requests.append({'candidate_id':r['candidate_id'],'request_sha256':req_hash,'request':request})
        if a.mode=='prepare': continue
        raw=''; error=None; result=None
        try:
            if a.mode=='mock':
                # Deliberately meaningless deterministic mock; never reads the target label.
                result={'form_prediction':LABELS[int(digest(json.dumps(model_input(r),sort_keys=True).encode())[:8],16)%2]}
                if cfg['condition']=='structured': result.update(typicality='uncertain',qualia='NONE',evidence='MOCK: no semantic inference')
                raw=json.dumps(result,ensure_ascii=False)
            else:
                proc=subprocess.run(a.adapter,input=encoded,text=True,capture_output=True,timeout=cfg['timeout_seconds'],check=True)
                raw=proc.stdout
            result=validate_response(json.loads(raw),cfg['condition'])
        except (ValueError,subprocess.SubprocessError) as exc:
            error=type(exc).__name__+': '+str(exc); result=None
        predictions.append({'candidate_id':r['candidate_id'],'prediction':result,'raw_response':raw,'error':error,'request_sha256':req_hash,'execution_mode':a.mode})
    write_rows(out/'requests.jsonl',requests)
    if a.mode!='prepare': write_rows(out/'predictions.jsonl',predictions)
    dump(out/'manifest.json',{'execution_mode':a.mode,'scientific_model_experiment_run':a.mode=='external',
        'model_id':a.model_id if a.mode=='external' else 'NO_MODEL_'+a.mode.upper(),'config':cfg,'config_sha256':digest(pathlib.Path(a.config).read_bytes()),
        'prompt_sha256':digest(prompt.encode()),'dataset_sha256':digest(pathlib.Path(a.dataset).read_bytes()),
        'demonstration_sha256':digest(pathlib.Path(a.demo_file).read_bytes()) if a.demo_file else None,
        'count':len(rows),'adapter_command':a.adapter,'code_sha256':digest(pathlib.Path(__file__).read_bytes()),
        'files':{f.name:digest(f.read_bytes()) for f in out.iterdir()}})
    print(f'{a.mode}: {len(rows)} requests; {out}')
if __name__=='__main__': main()

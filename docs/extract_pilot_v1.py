"""Conservative dependency extraction, stable IDs and immutable snapshots."""
import argparse, collections, csv, hashlib, json, pathlib, platform
VERSION = 'extract-v1.0.0'
LABELS = ('BN', 'YI_CL_N')
def digest(data):
    return hashlib.sha256(data).hexdigest()
def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf8')
def write_rows(path, rows):
    pathlib.Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows), encoding='utf8')
def read_rows(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf8').splitlines() if x.strip()]
def conllu(path):
    meta, tokens = {}, []
    lines = pathlib.Path(path).read_text(encoding='utf8').splitlines()+['']
    for line in lines:
        if not line.strip():
            if tokens:
                ids = {t['id'] for t in tokens}
                if len(ids)!=len(tokens) or any(t['head'] not in ids|{0} for t in tokens):
                    raise ValueError('Invalid token IDs/heads')
                for t in tokens:
                    seen, cur = set(), t
                    while cur['head']:
                        if cur['id'] in seen: raise ValueError('Dependency cycle')
                        seen.add(cur['id']); cur = next(x for x in tokens if x['id']==cur['head'])
                yield meta, tokens
            meta, tokens = {}, []
        elif line.startswith('#'):
            if ' = ' in line:
                k,v = line[2:].split(' = ',1); meta[k]=v
        else:
            f=line.split('\t')
            if len(f)!=10: raise ValueError('Expected ten CoNLL-U columns')
            if '-' in f[0] or '.' in f[0]: continue
            tokens.append(dict(zip(('id','form','lemma','upos','xpos','feats','head','rel','deps','misc'), f)))
            tokens[-1]['id']=int(f[0]); tokens[-1]['head']=int(f[6])
def extract(path, split, audit=None):
    audit = audit if audit is not None else collections.Counter()
    for index,(meta,ts) in enumerate(conllu(path),1):
        audit['sentences']+=1
        by={t['id']:t for t in ts}; children=collections.defaultdict(list)
        for t in ts: children[t['head']].append(t)
        for n in ts:
            if n['upos']!='NOUN' or n['xpos']=='NNB': continue
            audit['nouns']+=1
            v=by.get(n['head'])
            if not v or v['upos']!='VERB' or n['rel'].split(':')[0] not in ('obj','nsubj','obl','iobj'):
                audit['excluded_non_verbal_argument']+=1; continue
            ch=children[n['id']]
            nums=[t for t in ch if t['rel']=='nummod']
            dets=[t for t in ch if t['rel']=='det']
            yi=nums[0] if len(nums)==1 and nums[0]['form']=='一' else None
            clfs=([t for t in children[yi['id']] if t['rel']=='clf'] + [t for t in ch if t['rel']=='clf']) if yi else []
            if nums or dets or any(t['rel']=='clf' for t in ch):
                if yi and len(clfs)==1 and not dets and yi['id']<clfs[0]['id']<n['id']:
                    label='YI_CL_N'
                else: audit['excluded_determination']+=1; continue
            else: label='BN'
            # Remove entire argument subtree from model context, including all surface markers.
            subtree=set(); stack=[n['id']]
            while stack:
                i=stack.pop(); subtree.add(i); stack.extend(t['id'] for t in children[i])
            lo,hi=min(subtree),max(subtree)
            if subtree!=set(range(lo,hi+1)):
                audit['excluded_discontinuous_np']+=1; continue
            if v['id'] in subtree: raise ValueError('Predicate inside nominal subtree')
            sentence=meta.get('text',''.join(t['form'] for t in ts))
            sid=meta.get('sent_id',f'{split}-{index}')
            context=''.join('[TARGET_NP]' if t['id']==lo else '' if t['id'] in subtree else t['form'] for t in ts)
            flags=[]
            for rel in ('acl','nmod','conj','amod'):
                if any(t['rel'].split(':')[0]==rel for t in ch): flags.append('has_'+rel)
            row={'candidate_id':f'{split}:{sid}:{n["id"]}', 'source_split':split,'sentence_id':sid,
                 'sentence':sentence,'sentence_hash':digest(sentence.encode()),'noun':n['form'],'noun_lemma':n['lemma'],
                 'verb':v['form'],'verb_lemma':v['lemma'],'noun_token_id':n['id'],'verb_token_id':v['id'],
                 'relation':n['rel'],'observed_form':label,'classifier':clfs[0]['form'] if label=='YI_CL_N' else '',
                 'np_token_ids':sorted(subtree),'np_text':''.join(t['form'] for t in ts if t['id'] in subtree),
                 'masked_context':context,'flags':flags,'indefinite_status':'unreviewed','gold_typicality':None,'gold_qualia':None,
                 'extractor_version':VERSION,'dependency_info':ts}
            audit['accepted_'+label]+=1
            yield row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/ud'); p.add_argument('--out',required=True)
    p.add_argument('--pilot',action='store_true'); a=p.parse_args()
    out=pathlib.Path(a.out)
    if out.exists(): raise SystemExit('Refusing to overwrite snapshot; choose a new directory.')
    audits={}; rows=[]; sources={}
    for split in (('dev',) if a.pilot else ('train','dev','test')):
        path=pathlib.Path(a.data)/f'zh_gsdsimp-ud-{split}.conllu'; audit=collections.Counter()
        rows+=list(extract(path,split,audit)); audits[split]=dict(audit); sources[path.name]=digest(path.read_bytes())
    if a.pilot:
        # Review all available minority patterns before filling the 40-item pilot with BN.
        rows=sorted(rows,key=lambda r:(r['observed_form']=='BN',r['candidate_id']))[:40]
    rows.sort(key=lambda r:r['candidate_id'])
    assert len({r['candidate_id'] for r in rows})==len(rows)
    out.mkdir(parents=True)
    write_rows(out/'candidates.jsonl',rows)
    fields=[k for k in rows[0] if k!='dependency_info'] if rows else []
    with (out/'candidates.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
    split_hash={s:{r['sentence_hash'] for r in rows if r['source_split']==s} for s in ('train','dev','test')}
    lengths=sorted(len(r['dependency_info']) for r in rows)
    stats={'total':len(rows),'by_form':dict(collections.Counter(r['observed_form'] for r in rows)),
           'by_split_form':dict(collections.Counter(r['source_split']+'/'+r['observed_form'] for r in rows)),
           'unique_nouns':len({r['noun_lemma'] for r in rows}), 'unique_sentences':len({r['sentence_hash'] for r in rows}),
           'same_sentence_extra_occurrences':len(rows)-len({r['sentence_hash'] for r in rows}),
           'exact_occurrence_duplicates':len(rows)-len({(r['sentence_hash'],r['noun_token_id']) for r in rows}),
           'cross_split_sentence_overlap':{s+'/'+t:len(split_hash[s]&split_hash[t]) for s,t in [('train','dev'),('train','test'),('dev','test')]},
           'sentence_token_length':{'min':min(lengths,default=0),'max':max(lengths,default=0),'median':lengths[len(lengths)//2] if lengths else 0},
           'flags':dict(collections.Counter(x for r in rows for x in r['flags'])), 'audit':audits}
    dump(out/'statistics.json',stats)
    manifest={'version':VERSION,'created_date':'2026-10-07','python':platform.python_version(),'input_sha256':sources,
              'output_sha256':{f.name:digest(f.read_bytes()) for f in sorted(out.iterdir())},'label_status':'observed form, not indefinite gold',
              'sampling':'minority-first dev pilot, 40' if a.pilot else 'all eligible occurrences; no class rebalance',
              'code_sha256':digest(pathlib.Path(__file__).read_bytes()),'corpus_release':'unknown; exact uploaded bytes preserved'}
    dump(out/'manifest.json',manifest); print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

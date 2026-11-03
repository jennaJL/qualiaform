"""Frozen HF encoder + real PyTorch supervised heads and lexical baseline."""
import argparse,collections,hashlib,json,os,pathlib,time
import numpy as np
from .extract import read_rows,write_rows,dump,digest,LABELS
from .evaluate import metrics
ROOT=pathlib.Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'configs/portfolio_protocol_v1.json'
DATASET=ROOT/'data/processed/ud_candidates_v1/candidates.jsonl'
def sha_file(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def input_text(row):
    return f"名词：{row['noun_lemma']}；动词：{row['verb_lemma']}；关系：{row['relation']}；上下文：{row['masked_context']}"
def encoder_path(override=None):
    cfg=json.loads(PROTOCOL.read_text())
    options=[override,os.environ.get('QUALIAFORM_ENCODER'),ROOT/'models/minilm_base',pathlib.Path.home()/'.cache/huggingface/hub'/('models--'+cfg['encoder_repo'].replace('/','--'))/'snapshots'/cfg['encoder_revision']]
    for p in options:
        if p and (pathlib.Path(p)/'model.safetensors').exists():return pathlib.Path(p).resolve()
    raise FileNotFoundError('Pinned MiniLM weights not found; run scripts/setup_portfolio.py or set QUALIAFORM_ENCODER')
class Encoder:
    def __init__(self,path=None):
        import torch
        from transformers import AutoTokenizer,AutoModel
        self.torch=torch;torch.set_num_threads(4);self.path=encoder_path(path)
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False).eval()
        self.max_length=json.loads(PROTOCOL.read_text())['max_length']
    def encode(self,rows):
        torch=self.torch;texts=[input_text(r) for r in rows]
        encoded=self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
        with torch.inference_mode():
            h=self.model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
            features=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            features=torch.nn.functional.normalize(features,p=2,dim=1)
        return features.cpu().numpy()
def features(a):
    cfg=json.loads(PROTOCOL.read_text());out=pathlib.Path(a.out)
    if out.exists():raise ValueError('Feature snapshot exists; use a new directory')
    out.mkdir(parents=True);rows=read_rows(DATASET);enc=Encoder(a.encoder);start=time.monotonic();xs=[]
    for i in range(0,len(rows),cfg['encoder_batch_size']):
        xs.append(enc.encode(rows[i:i+cfg['encoder_batch_size']]))
        if i%320==0:print(f'Encoded {min(i+32,len(rows))}/{len(rows)}',flush=True)
    x=np.concatenate(xs);np.savez_compressed(out/'features.npz',x=x,ids=np.array([r['candidate_id'] for r in rows]))
    names=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors','1_Pooling/config.json','sentence_bert_config.json']
    enc_manifest={'repo':cfg['encoder_repo'],'revision':cfg['encoder_revision'],'files':{n:{'sha256':sha_file(enc.path/n),'bytes':(enc.path/n).stat().st_size} for n in names if (enc.path/n).is_file()}}
    dump(ROOT/'configs/encoder_manifest.json',enc_manifest)
    lengths=[len(enc.tokenizer.encode(input_text(r))) for r in rows]
    dump(out/'manifest.json',{'dataset_sha256':sha_file(DATASET),'protocol_sha256':sha_file(PROTOCOL),'encoder_manifest_sha256':sha_file(ROOT/'configs/encoder_manifest.json'),
       'features_sha256':sha_file(out/'features.npz'),'count':len(rows),'dimensions':x.shape[1],'truncated_inputs':sum(n>cfg['max_length'] for n in lengths),
       'elapsed_seconds':time.monotonic()-start,'label_fields_used':False})
    print('FEATURES COMPLETE',x.shape,flush=True)
def head(kind,dim,hidden=128,dropout=.1):
    import torch.nn as nn
    return nn.Linear(dim,2) if kind=='linear' else nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,2))
def choose_threshold(y,probs):
    candidates=[]
    for t in np.linspace(.005,.995,199):
        m=metrics([(LABELS[int(a)],LABELS[int(b>=t)]) for a,b in zip(y,probs)])
        candidates.append((m['macro_f1'],m['balanced_accuracy'] or 0,-abs(float(t)-.5),float(t)))
    best=max(candidates);return best[-1],best[0]
def score(y,prob,threshold):
    return metrics([(LABELS[int(a)],LABELS[int(p>=threshold)]) for a,p in zip(y,prob)])
def bootstrap(rows,prob,threshold,repeats=500):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['sentence_hash']].append(i)
    arrays=list(groups.values());rng=np.random.default_rng(1729);values=[]
    for _ in range(repeats):
        ix=[i for j in rng.integers(0,len(arrays),len(arrays)) for i in arrays[j]]
        m=metrics([(rows[i]['observed_form'],LABELS[int(prob[i]>=threshold)]) for i in ix])
        if m['balanced_accuracy'] is not None:values.append([m['balanced_accuracy'],m['macro_f1']])
    q=np.quantile(np.array(values),[.025,.975],axis=0)
    return {'resampling_unit':'sentence','resamples':len(values),'balanced_accuracy_95':[float(q[0,0]),float(q[1,0])],'macro_f1_95':[float(q[0,1]),float(q[1,1])], 'interpretation':'descriptive uncertainty on this reused weak-label benchmark; not significance evidence'}
def train(a):
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    from safetensors.torch import save_file
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(4);cfg=json.loads(PROTOCOL.read_text());out=pathlib.Path(a.out)
    if out.exists():raise ValueError('Benchmark snapshot exists; use a new directory')
    fdir=pathlib.Path(a.features);fm=json.loads((fdir/'manifest.json').read_text());assert fm['dataset_sha256']==sha_file(DATASET) and fm['protocol_sha256']==sha_file(PROTOCOL)
    assert fm['features_sha256']==sha_file(fdir/'features.npz')
    data=np.load(fdir/'features.npz',allow_pickle=False);rows=read_rows(DATASET);assert list(data['ids'])==[r['candidate_id'] for r in rows]
    x=data['x'];y=np.array([LABELS.index(r['observed_form']) for r in rows]);ix={s:np.array([i for i,r in enumerate(rows) if r['source_split']==s]) for s in ('train','dev','test')}
    assert not ({rows[i]['sentence_hash'] for i in ix['train']}&{rows[i]['sentence_hash'] for i in ix['dev']})
    assert not ({rows[i]['sentence_hash'] for i in ix['train']}&{rows[i]['sentence_hash'] for i in ix['test']})
    mean=x[ix['train']].mean(0);std=x[ix['train']].std(0).clip(.01);z=torch.from_numpy((x-mean)/std);yt=torch.from_numpy(y)
    out.mkdir(parents=True);dump(out/'protocol.json',cfg);counts=np.bincount(y[ix['train']],minlength=2);weights=torch.tensor(np.sqrt(counts.max()/counts),dtype=torch.float32)
    history=[];models=[];start=time.monotonic()
    for kind in cfg['models']:
        for seed in cfg['seeds']:
            torch.manual_seed(seed);np.random.seed(seed);m=head(kind,z.shape[1],cfg['mlp_hidden'],cfg['dropout'])
            optimizer=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);criterion=torch.nn.CrossEntropyLoss(weight=weights)
            loader=DataLoader(TensorDataset(z[ix['train']],yt[ix['train']]),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            for epoch in range(cfg['epochs']):
                m.train();total=0.;n=0
                for xb,yb in loader:
                    optimizer.zero_grad(set_to_none=True);loss=criterion(m(xb),yb)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(yb);n+=len(yb)
                history.append({'model':kind,'seed':seed,'epoch':epoch+1,'train_loss':total/n})
            m.eval()
            with torch.inference_mode():p=m(z[ix['dev']]).softmax(-1)[:,1].numpy()
            threshold,dev_f1=choose_threshold(y[ix['dev']],p)
            tag=f'{kind}_{seed}';folder=out/tag;folder.mkdir();state={k:v.contiguous() for k,v in m.state_dict().items()};state['feature_mean']=torch.from_numpy(mean);state['feature_std']=torch.from_numpy(std)
            save_file(state,str(folder/'model.safetensors'))
            meta={'kind':kind,'seed':seed,'input_dim':z.shape[1],'hidden':cfg['mlp_hidden'],'dropout':cfg['dropout'],'threshold':threshold,'score_interpretation':'uncalibrated YI score',
                  'dev':score(y[ix['dev']],p,threshold),'parameters':sum(t.numel() for t in m.parameters()),'encoder_frozen':True,'weights_sha256':sha_file(folder/'model.safetensors')}
            dump(folder/'model.json',meta);models.append((tag,m,meta));print(tag,'dev_macro_f1',dev_f1,'threshold',threshold,flush=True)
    # Serving model selection is sealed using dev, before any new model's test scoring.
    eligible=[v for v in models if v[2]['seed']==cfg['seeds'][0]];champion=max(eligible,key=lambda v:v[2]['dev']['macro_f1']);dump(out/'selection.json',{'selected':champion[0],'criterion':'dev macro-F1, primary seed 1729; test not consulted','dev_macro_f1':champion[2]['dev']['macro_f1']})
    text=[input_text(r) for r in rows];vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=20000,sublinear_tf=True)
    tx=vectorizer.fit_transform([text[i] for i in ix['train']]);lex=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=1729);lex.fit(tx,y[ix['train']]);dp=lex.predict_proba(vectorizer.transform([text[i] for i in ix['dev']]))[:,1];lt,_=choose_threshold(y[ix['dev']],dp)
    dump(out/'lexical_model.json',{'threshold':lt,'vocabulary':{k:int(v) for k,v in vectorizer.vocabulary_.items()},'dev':score(y[ix['dev']],dp,lt),'configuration':cfg['lexical_baseline']})
    np.savez_compressed(out/'lexical_weights.npz',idf=vectorizer.idf_,coef=lex.coef_,intercept=lex.intercept_)
    testrows=[rows[i] for i in ix['test']];result={}
    with torch.inference_mode():
        for tag,m,meta in models:
            pred=m(z[ix['test']]).softmax(-1)[:,1].numpy();result[tag]=score(y[ix['test']],pred,meta['threshold'])
            write_rows(out/tag/'test_predictions.jsonl',[{'candidate_id':r['candidate_id'],'observed_form':r['observed_form'],'prediction':LABELS[int(p>=meta['threshold'])],'yi_score_uncalibrated':float(p)} for r,p in zip(testrows,pred)])
            if tag==champion[0]:selected_probs=pred
    lp=lex.predict_proba(vectorizer.transform([text[i] for i in ix['test']]))[:,1];result['char_tfidf_logreg']=score(y[ix['test']],lp,lt);result['majority_BN']=score(y[ix['test']],np.zeros(len(testrows)),.5)
    write_rows(out/'lexical_test_predictions.jsonl',[{'candidate_id':r['candidate_id'],'observed_form':r['observed_form'],'prediction':LABELS[int(p>=lt)],'yi_score_uncalibrated':float(p)} for r,p in zip(testrows,lp)])
    dump(out/'test_metrics.json',result);write_rows(out/'training_history.jsonl',history)
    dump(out/'selected_bootstrap.json',bootstrap(testrows,selected_probs,champion[2]['threshold']))
    slices={}
    train_pairs={(rows[i]['verb_lemma'],rows[i]['noun_lemma']) for i in ix['train']}
    for label,mask in [('seen_VN',[(r['verb_lemma'],r['noun_lemma']) in train_pairs for r in testrows]),('unseen_VN',[(r['verb_lemma'],r['noun_lemma']) not in train_pairs for r in testrows]),('contains_proper_name',['contains_proper_name' in r['flags'] for r in testrows])]:
        idx=np.where(mask)[0];slices[label]=score(y[ix['test']][idx],selected_probs[idx],champion[2]['threshold'])
    dump(out/'selected_slices.json',slices)
    errors=[{**r,'prediction':LABELS[int(p>=champion[2]['threshold'])],'yi_score_uncalibrated':float(p),'error_type':'unreviewed'} for r,p in zip(testrows,selected_probs) if r['observed_form']!=LABELS[int(p>=champion[2]['threshold'])]]
    write_rows(out/'errors.jsonl',errors)
    dump(out/'manifest.json',{'status':'COMPLETED','created_date':'2026-10-08','dataset_sha256':sha_file(DATASET),'feature_sha256':sha_file(fdir/'features.npz'),'protocol_sha256':sha_file(PROTOCOL),
      'code_sha256':sha_file(__file__),'train_n':len(ix['train']),'dev_n':len(ix['dev']),'test_n':len(ix['test']),'elapsed_seconds':time.monotonic()-start,'selected':champion[0],'gradient_updates':'classification heads only; encoder frozen',
      'test_reuse_notice':cfg['test_policy'],'reference':'weak observed-form labels; not expert indefinite gold','paid_api_calls':0})
    print('BENCHMARK COMPLETE',champion[0],result[champion[0]],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['features','train']);p.add_argument('--encoder');p.add_argument('--features',default=str(ROOT/'data/portfolio_features_v1'));p.add_argument('--out',required=True);a=p.parse_args();features(a) if a.command=='features' else train(a)
if __name__=='__main__':main()

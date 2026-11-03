"""Real, offline CPU inference and LoRA. No API credentials or paid services."""
import argparse,datetime,hashlib,json,os,pathlib,time
from .extract import read_rows,write_rows,dump,digest
from .experiment import model_input,validate_response
ROOT=pathlib.Path(__file__).resolve().parents[2]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def messages(request):
    result=[{'role':'system','content':request['system']}]
    for d in request['demonstrations']:
        result.extend([{'role':'user','content':json.dumps(d['input'],ensure_ascii=False)}, {'role':'assistant','content':json.dumps(d['output'],ensure_ascii=False)}])
    result.append({'role':'user','content':json.dumps(request['input'],ensure_ascii=False)})
    return result

def load_runtime(path,adapter=None):
    # These imports are intentionally optional: original offline scaffold remains dependency-free.
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(1729)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True)
    model.eval()
    return torch,model,tok

def parse_response(raw,condition):
    s=raw.strip()
    if s.startswith('```json\n') and s.endswith('```'):s=s[8:-3].strip()
    elif s.startswith('```\n') and s.endswith('```'):s=s[4:-3].strip()
    return validate_response(json.loads(s),condition)

def requests_for(rows,condition,shots):
    prompt=(ROOT/f'prompts/local_v2/{condition}.txt').read_text()
    demos=[]
    if shots=='corpus_few':
        for d in read_rows(ROOT/'data/local_llm_v1/corpus_demos.jsonl'):
            if d['source_split']!='train':raise ValueError('non-train demonstration')
            if any(d['sentence_hash']==r['sentence_hash'] or d['input']['masked_context']==r['masked_context'] for r in rows):raise ValueError('demo/evaluation overlap')
            demos.append({'input':d['input'],'output':d['output'] if condition=='structured' else {'form_prediction':d['output']['form_prediction']}})
    result=[]
    for r in rows:
        req={'system':prompt,'demonstrations':demos,'input':model_input(r),'generation':{'do_sample':False,'seed':1729,'max_new_tokens':64,'decoder':'finite_json_enum_v1'}}
        result.append({'candidate_id':r['candidate_id'],'request_sha256':digest(json.dumps(req,ensure_ascii=False,sort_keys=True).encode()),'request':req})
    return result

def response_candidates(condition):
    if condition=='direct':return [{'form_prediction':label} for label in ('BN','YI_CL_N')]
    return [{'typicality':t,'qualia':q,'form_prediction':label,'evidence':''} for t in ('typical','atypical','uncertain') for q in ('TELIC','AGENTIVE','HAVE','NONE') for label in ('BN','YI_CL_N')]

def constrained_decoder(tokenizer,condition,prefix_length):
    # Model logits choose tokens within the schema; no reference labels are accessible here.
    trie={}
    for obj in response_candidates(condition):
        ids=tokenizer.encode(json.dumps(obj,ensure_ascii=False,separators=(',',':')),add_special_tokens=False)+[tokenizer.eos_token_id]
        node=trie
        for token in ids:node=node.setdefault(token,{})
    def allowed(batch_id,input_ids):
        node=trie
        for token in input_ids.tolist()[prefix_length:]:
            if token==tokenizer.eos_token_id:return [tokenizer.eos_token_id]
            node=node[token]
        return list(node) or [tokenizer.eos_token_id]
    return allowed

def infer(a):
    dataset=ROOT/'data/processed/ud_candidates_v1/candidates.jsonl'
    allrows=read_rows(dataset)
    rows=read_rows(ROOT/'data/local_llm_v1/pilot_rows.jsonl') if a.split=='pilot' else [r for r in allrows if r['source_split']==a.split]
    if a.limit: rows=rows[:a.limit]
    reqs=requests_for(rows,a.condition,a.shots)
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model_manifest=json.loads((ROOT/'configs/cpu_model_manifest.json').read_text())
    identity={'condition':a.condition,'shots':a.shots,'split':a.split,'count':len(rows),'batch_size':a.batch_size,
      'model_revision':model_manifest['revision'],'adapter':str(pathlib.Path(a.adapter).resolve()) if a.adapter else None,
      'adapter_sha256':digest((pathlib.Path(a.adapter)/'adapter_model.safetensors').read_bytes()) if a.adapter else None,
      'dataset_sha256':digest(dataset.read_bytes()),'requests_sha256':digest(json.dumps(reqs,ensure_ascii=False,sort_keys=True).encode()),
      'code_sha256':digest(pathlib.Path(__file__).read_bytes())}
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Resume identity mismatch')
        previous=read_rows(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else []
        if len(previous)!=len({p['candidate_id'] for p in previous}):raise ValueError('Duplicate predictions')
        if [p['candidate_id'] for p in previous]!=[r['candidate_id'] for r in reqs[:len(previous)]]:raise ValueError('Non-prefix partial run')
    else:
        dump(out/'identity.json',identity);write_rows(out/'requests.jsonl',reqs);previous=[]
    done=len(previous)
    if done==len(rows) and (out/'manifest.json').exists():print('Already complete',out);return
    torch,model,tok=load_runtime(a.model,a.adapter)
    start=time.monotonic()
    with (out/'predictions.jsonl').open('a',encoding='utf8') as f:
        for offset in range(done,len(reqs),a.batch_size):
            batch=reqs[offset:offset+a.batch_size]
            texts=[tok.apply_chat_template(messages(r['request']),tokenize=False,add_generation_prompt=True) for r in batch]
            inp=tok(texts,return_tensors='pt',padding=True,truncation=False)
            t=time.monotonic()
            allowed=constrained_decoder(tok,a.condition,inp['input_ids'].shape[1])
            with torch.inference_mode():
                generated=model.generate(**inp,do_sample=False,max_new_tokens=batch[0]['request']['generation']['max_new_tokens'],pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,prefix_allowed_tokens_fn=allowed)
            outputs=tok.batch_decode(generated[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
            for req,raw in zip(batch,outputs):
                try:obj=parse_response(raw,a.condition);err=None
                except (ValueError,TypeError) as exc:obj=None;err=str(exc)
                pred={'candidate_id':req['candidate_id'],'prediction':obj,'raw_response':raw,'error':err,'request_sha256':req['request_sha256'],'execution_mode':'external','backend':'local_cpu','generated_at':now()}
                f.write(json.dumps(pred,ensure_ascii=False,sort_keys=True)+'\n')
            f.flush();os.fsync(f.fileno())
            print(f'{a.condition}/{a.shots} {offset+len(batch)}/{len(reqs)} batch_seconds={time.monotonic()-t:.1f}',flush=True)
    manifest={'execution_mode':'external','scientific_model_experiment_run':True,'backend':'local_cpu','paid_api_calls':0,
       'model_id':model_manifest['repo'],'model_revision':model_manifest['revision'],'config':identity,'config_sha256':digest(json.dumps(identity,sort_keys=True).encode()),
       'dataset_sha256':identity['dataset_sha256'],'count':len(rows),'prompt_sha256':digest(reqs[0]['request']['system'].encode()),
       'demonstration_sha256':digest((ROOT/'data/local_llm_v1/corpus_demos.jsonl').read_bytes()) if a.shots=='corpus_few' else None,
       'semantic_demo_provenance':'assistant provisional annotations on UD train, not Part1 gold' if a.shots=='corpus_few' else None,
       'elapsed_this_invocation_seconds':time.monotonic()-start,'completed_at':now(),'files':{p.name:digest(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
    dump(out/'manifest.json',manifest)

def train(a):
    import random
    from peft import LoraConfig,get_peft_model,TaskType
    out=pathlib.Path(a.out)
    if out.exists():raise ValueError('Refusing to overwrite training')
    protocol=json.loads((ROOT/'configs/local_protocol_v2.json').read_text());cfg=protocol['training']
    torch,base,tok=load_runtime(a.model)
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],target_modules=cfg['target_modules'],layers_to_transform=cfg['layers'],bias='none'))
    model.config.use_cache=False
    prompt=(ROOT/'prompts/local_v2/direct.txt').read_text()
    def encode(r):
        msgs=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(model_input(r),ensure_ascii=False)}]
        prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
        full=tok.apply_chat_template(msgs+[{'role':'assistant','content':json.dumps({'form_prediction':r['observed_form']})}],tokenize=True)
        assert full[:len(prefix)]==prefix
        if len(full)>cfg['max_length']:raise ValueError('Training sample exceeds frozen length limit')
        return full,[-100]*len(prefix)+full[len(prefix):]
    trainrows=read_rows(ROOT/'data/local_llm_v1/train_rows.jsonl');validrows=read_rows(ROOT/'data/local_llm_v1/dev_rows.jsonl')
    assert all(r['source_split']=='train' for r in trainrows) and all(r['source_split']=='dev' for r in validrows)
    trainset=[encode(r) for r in trainrows];validset=[encode(r) for r in validrows]
    def batch(data):
        length=max(len(x) for x,y in data)
        ids=torch.tensor([[tok.pad_token_id]*(length-len(x))+x for x,y in data]);labels=torch.tensor([[-100]*(length-len(y))+y for x,y in data])
        attention=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x,y in data])
        return ids,attention,labels
    def loss(data):
        ids,attention,labels=batch(data)
        # All answer spans fit in the final 32 tokens; avoid a full-vocabulary logit tensor for every prompt token.
        assert (labels[:,:-32]==-100).all()
        outputs=model(input_ids=ids,attention_mask=attention,logits_to_keep=33)
        logits=outputs.logits[:,:-1,:];target=labels[:,-32:]
        return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100)
    def validate():
        model.eval();values=[]
        with torch.no_grad():
            for i in range(0,len(validset),4):values.append(float(loss(validset[i:i+4])))
        return sum(values)/len(values)
    out.mkdir(parents=True)
    dump(out/'protocol.json',protocol)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    start=time.monotonic();before=validate();print(f'initial_validation_loss={before:.6f}',flush=True)
    rng=random.Random(1729);rng.shuffle(trainset);history=[];model.train()
    for offset in range(0,len(trainset),cfg['batch_size']):
        opt.zero_grad(set_to_none=True);v=loss(trainset[offset:offset+cfg['batch_size']]);v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        row={'step':offset//cfg['batch_size']+1,'loss':float(v.detach()),'elapsed_seconds':time.monotonic()-start};history.append(row)
        print(json.dumps(row),flush=True)
    after=validate();model.save_pretrained(out,safe_serialization=True)
    changed=sum(int(not torch.equal(initial[n],p.detach())) for n,p in model.named_parameters() if p.requires_grad)
    write_rows(out/'training_history.jsonl',history)
    dump(out/'training_result.json',{'status':'COMPLETED','actual_weight_updates':True,'changed_parameter_tensors':changed,
       'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'steps':len(history),'train_examples':len(trainset),
       'validation_examples':len(validset),'validation_loss_before':before,'validation_loss_after':after,'elapsed_seconds':time.monotonic()-start,
       'paid_api_calls':0,'training_reference':'weak observed-form labels, not indefinite gold','protocol_sha256':digest((out/'protocol.json').read_bytes()),
       'model_manifest_sha256':digest((ROOT/'configs/cpu_model_manifest.json').read_bytes()),'adapter_sha256':digest((out/'adapter_model.safetensors').read_bytes()),
       'training_input_sha256':digest((ROOT/'data/local_llm_v1/train_rows.jsonl').read_bytes()),'validation_input_sha256':digest((ROOT/'data/local_llm_v1/dev_rows.jsonl').read_bytes()),
       'code_sha256':digest(pathlib.Path(__file__).read_bytes())})
    print('TRAINING COMPLETE',changed,'parameter tensors changed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['infer','train']);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--condition',choices=['direct','structured'],default='direct');p.add_argument('--shots',choices=['zero','corpus_few'],default='zero')
    p.add_argument('--split',choices=['pilot','dev','test'],default='test');p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--adapter')
    a=p.parse_args();train(a) if a.command=='train' else infer(a)
if __name__=='__main__':main()

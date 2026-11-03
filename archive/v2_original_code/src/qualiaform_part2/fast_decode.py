"""Greedy finite-schema decoding: skip model passes at single-choice syntax nodes.

At each genuine branch, re-prefill the complete chosen prefix and select the highest
model logit among allowed tokens. Forced syntax tokens are unchanged. No labels or
reference data enter this decoder. This trades KV cache reuse for fewer forwards.
"""
def generate(model,torch,input_ids,attention_mask,*,max_new_tokens,prefix_allowed_tokens_fn,pad_token_id,eos_token_id,**kwargs):
    if kwargs.get('do_sample',False):raise ValueError('Fast decoder supports greedy only')
    bases=input_ids.tolist();masks=attention_mask.tolist();generated=[[] for _ in bases];done=[False]*len(bases)
    while not all(done):
        branching=[]
        for i in range(len(bases)):
            if done[i]:continue
            while len(generated[i])<max_new_tokens:
                allowed=sorted(prefix_allowed_tokens_fn(i,torch.tensor(bases[i]+generated[i])))
                if len(allowed)>1:
                    branching.append((i,allowed));break
                if not allowed:raise ValueError('Empty grammar node')
                token=allowed[0];generated[i].append(token)
                if token==eos_token_id:done[i]=True;break
            if len(generated[i])>=max_new_tokens:done[i]=True
        if not branching:continue
        length=max(len(bases[i])+len(generated[i]) for i,_ in branching)
        ids=[];att=[]
        for i,_ in branching:
            seq=bases[i]+generated[i];pad=length-len(seq)
            ids.append([pad_token_id]*pad+seq);att.append([0]*pad+masks[i]+[1]*len(generated[i]))
        ids=torch.tensor(ids,dtype=input_ids.dtype);att=torch.tensor(att,dtype=attention_mask.dtype)
        positions=att.long().cumsum(-1)-1;positions.masked_fill_(att==0,1)
        logits=model(input_ids=ids,attention_mask=att,position_ids=positions,use_cache=False,logits_to_keep=1).logits[:,-1,:]
        for j,(i,allowed) in enumerate(branching):
            token=allowed[int(torch.argmax(logits[j,allowed]))];generated[i].append(token)
            if token==eos_token_id or len(generated[i])>=max_new_tokens:done[i]=True
    length=max(len(g) for g in generated)
    return torch.tensor([base+g+[pad_token_id]*(length-len(g)) for base,g in zip(bases,generated)],dtype=input_ids.dtype)

"""Freeze corpus-only demonstrations and a small train-only LoRA protocol."""
import pathlib,sys,collections,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from qualiaform_part2.extract import read_rows,write_rows,dump,digest
from qualiaform_part2.experiment import model_input
root=pathlib.Path(__file__).resolve().parents[1];out=root/'data/local_llm_v1'
rows=read_rows(root/'data/processed/ud_candidates_v1/candidates.jsonl');by={r['candidate_id']:r for r in rows}
choices=[('train:train-s1866:12','TELIC','牙线用于清洁牙缝'),('train:train-s1134:13','AGENTIVE','写信形成文本'),('train:train-s1505:11','AGENTIVE','书写形成公式文本'),('train:train-s2261:21','AGENTIVE','建立机关使其形成')]
demos=[]
for cid,qualia,evidence in choices:
 r=by[cid];demos.append({'candidate_id':cid,'sentence_hash':r['sentence_hash'],'provenance':'ud_train_observed_form_plus_assistant_semantic_annotation',
 'source_split':'train','semantic_annotation_status':'provisional_not_expert_gold','input':model_input(r),
 'output':{'form_prediction':r['observed_form'],'typicality':'typical','qualia':qualia,'evidence':evidence}})
write_rows(out/'corpus_demos.jsonl',demos)
selected={}
for split,n in [('train',128),('dev',20)]:
 selected[split]=[]
 for label in ('BN','YI_CL_N'):
  pool=[r for r in rows if r['source_split']==split and r['observed_form']==label]
  selected[split]+=sorted(pool,key=lambda r:digest(('lora-v1:'+r['candidate_id']).encode()))[:n]
 selected[split].sort(key=lambda r:digest(('order-v1:'+r['candidate_id']).encode()))
 write_rows(out/(split+'_rows.jsonl'),selected[split])
# Dev pilot shares no role as independent test; first stable 10 per class.
pilot=[]
for label in ('BN','YI_CL_N'):
 pilot+=sorted([r for r in rows if r['source_split']=='dev' and r['observed_form']==label],key=lambda r:r['candidate_id'])[:10]
write_rows(out/'pilot_rows.jsonl',pilot)
protocol={'name':'local-cpu-v1','created_date':'2026-10-07','model':'Qwen/Qwen2.5-0.5B-Instruct','device':'cpu','precision':'float32',
 'seed':1729,'inference':{'batch_size':4,'do_sample':False,'max_new_tokens_direct':32,'max_new_tokens_structured':96},
 'fewshot':'4 source-train corpus examples; observed labels; assistant provisional semantics; NOT Part1 gold',
 'training':{'method':'LoRA','rank':4,'alpha':8,'dropout':0,'layers':[20,21,22,23],'target_modules':['q_proj','v_proj'],
 'examples':256,'class_counts':{'BN':128,'YI_CL_N':128},'epochs':1,'batch_size':4,'learning_rate':0.0002,'max_length':768,'validation_examples':40,'checkpoint_selection':'final epoch, no test-based selection'},
 'evaluation':'complete original test split, 913 examples, weak observed-form reference; no test tuning',
 'files':{f.name:digest(f.read_bytes()) for f in out.glob('*.jsonl')}}
dump(root/'configs/local_protocol_v1.json',protocol)
print('Frozen local experiment and LoRA protocol; 256 train / 40 validation / 4 corpus demos.')

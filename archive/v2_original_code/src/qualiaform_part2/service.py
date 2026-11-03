"""Local FastAPI inference, train-only example retrieval, and health telemetry."""
import contextlib,json,logging,os,pathlib,threading,time,uuid
from typing import Annotated,Literal
import numpy as np
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,PlainTextResponse
from pydantic import BaseModel,ConfigDict,Field,field_validator
from .portfolio import ROOT,Encoder,head,input_text,sha_file
from .extract import read_rows,LABELS
LOG=logging.getLogger('qualiaform.requests')
NOTICE='Predicts extracted observed form, not indefinite eligibility. Scores are uncalibrated; expert review is required.'
class PredictInput(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    masked_context:Annotated[str,Field(min_length=1,max_length=1200)]
    noun_lemma:Annotated[str,Field(min_length=1,max_length=80)]
    verb_lemma:Annotated[str,Field(min_length=1,max_length=80)]
    relation:Annotated[str,Field(pattern=r'^(obj|iobj|nsubj|obl)(:[a-z]+)?$')]='obj'
    @field_validator('masked_context')
    @classmethod
    def single_target(cls,v):
        if v.count('[TARGET_NP]')!=1:raise ValueError('Use exactly one [TARGET_NP] marker')
        return v
class BatchInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    items:Annotated[list[PredictInput],Field(min_length=1,max_length=16)]
class Prediction(BaseModel):
    label:Literal['BN','YI_CL_N']
    yi_score_uncalibrated:float
    decision_threshold:float
    model_id:str
    input_truncated:bool
    nearest_training_examples:list[dict]
    notice:str=NOTICE
class ModelService:
    def __init__(self,benchmark=None,encoder=None):
        import torch
        from safetensors.torch import load_file
        self.torch=torch;self.benchmark=pathlib.Path(benchmark or os.environ.get('QUALIAFORM_BENCHMARK',ROOT/'results/portfolio_v1'))
        self.selection=json.loads((self.benchmark/'selection.json').read_text());self.name=self.selection['selected'];folder=self.benchmark/self.name
        self.meta=json.loads((folder/'model.json').read_text())
        if sha_file(folder/'model.safetensors')!=self.meta['weights_sha256']:raise ValueError('Classifier checksum mismatch')
        self.encoder=Encoder(encoder)
        manifest=json.loads((ROOT/'configs/encoder_manifest.json').read_text())
        for name,info in manifest['files'].items():
            if sha_file(self.encoder.path/name)!=info['sha256']:raise ValueError('Encoder checksum mismatch: '+name)
        state=load_file(str(folder/'model.safetensors'));self.mean=state.pop('feature_mean');self.std=state.pop('feature_std')
        self.model=head(self.meta['kind'],self.meta['input_dim'],self.meta['hidden'],self.meta['dropout']);self.model.load_state_dict(state);self.model.eval()
        fdir=ROOT/'data/portfolio_features_v1';fm=json.loads((fdir/'manifest.json').read_text())
        if sha_file(fdir/'features.npz')!=fm['features_sha256']:raise ValueError('Feature cache checksum mismatch')
        data=np.load(fdir/'features.npz',allow_pickle=False)
        rows=read_rows(ROOT/'data/processed/ud_candidates_v1/candidates.jsonl')
        if list(data['ids'])!=[r['candidate_id'] for r in rows]:raise ValueError('Feature/row ID mismatch')
        idx=[i for i,r in enumerate(rows) if r['source_split']=='train']
        self.train_features=data['x'][idx];fields=['candidate_id','sentence','masked_context','noun_lemma','verb_lemma','relation','observed_form']
        self.train_rows=[{k:rows[i][k] for k in fields} for i in idx];self.lock=threading.Lock()
    def predict(self,inputs):
        rows=[r.model_dump() if hasattr(r,'model_dump') else r for r in inputs]
        with self.lock:
            x=self.encoder.encode(rows)
            truncation_flags=[len(self.encoder.tokenizer.encode(input_text(r)))>self.encoder.max_length for r in rows]
            with self.torch.inference_mode():scores=self.model((self.torch.from_numpy(x)-self.mean)/self.std).softmax(-1)[:,1].numpy()
        results=[]
        for r,v,p,truncated in zip(rows,x,scores,truncation_flags):
            sims=self.train_features@v;top=np.argsort(-sims)[:3]
            examples=[{**self.train_rows[i],'cosine_similarity':float(sims[i]),'source_split':'train'} for i in top]
            results.append({'label':LABELS[int(p>=self.meta['threshold'])],'yi_score_uncalibrated':float(p),'decision_threshold':self.meta['threshold'],
              'model_id':self.name,'input_truncated':truncated,
              'nearest_training_examples':examples,'notice':NOTICE})
        return results
    def benchmark_summary(self):
        return {'selected':self.name,'selection_basis':'development macro-F1, seed 1729','test_n':913,'reference':'weak observed form',
                'test_reused':True,'models':json.loads((self.benchmark/'test_metrics.json').read_text()),'bootstrap':json.loads((self.benchmark/'selected_bootstrap.json').read_text())}
def create_app(service=None):
    stats={'requests':0,'errors':0,'seconds':0.};lock=threading.Lock()
    @contextlib.asynccontextmanager
    async def lifespan(app):
        app.state.model=service or ModelService();yield
    app=FastAPI(title='QualiaForm Research API',version='2.0.0',description=NOTICE,lifespan=lifespan)
    @app.middleware('http')
    async def telemetry(request:Request,call_next):
        start=time.perf_counter();rid=str(uuid.uuid4());status=500
        try:
            response=await call_next(request);status=response.status_code;response.headers['X-Request-ID']=rid;return response
        finally:
            elapsed=time.perf_counter()-start
            with lock:stats['requests']+=1;stats['errors']+=int(status>=400);stats['seconds']+=elapsed
            # Never log user text or model inputs.
            LOG.info(json.dumps({'request_id':rid,'method':request.method,'path':request.url.path,'status':status,'seconds':round(elapsed,6)}))
    @app.get('/',response_class=HTMLResponse,include_in_schema=False)
    def page():return (pathlib.Path(__file__).parent/'static/index.html').read_text(encoding='utf8')
    @app.get('/healthz')
    def health():return {'status':'ready','model':app.state.model.name,'backend':'local_cpu','reference':'weak_observed_form'}
    @app.post('/predict',response_model=Prediction)
    def predict(item:PredictInput):return app.state.model.predict([item])[0]
    @app.post('/predict_batch',response_model=list[Prediction])
    def predict_batch(batch:BatchInput):return app.state.model.predict(batch.items)
    @app.get('/benchmark')
    def benchmark():return app.state.model.benchmark_summary()
    @app.get('/metrics',response_class=PlainTextResponse)
    def monitoring():
        with lock:s=dict(stats)
        return f"# TYPE qualiaform_requests_total counter\nqualiaform_requests_total {s['requests']}\n# TYPE qualiaform_errors_total counter\nqualiaform_errors_total {s['errors']}\n# TYPE qualiaform_request_seconds_sum counter\nqualiaform_request_seconds_sum {s['seconds']:.6f}\n"
    return app

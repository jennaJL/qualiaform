"""Launch local demo only; Ctrl-C stops it."""
import os,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
if __name__=='__main__':
 import uvicorn
 uvicorn.run('qualiaform_part2.service:create_app',factory=True,host='127.0.0.1',port=8765)

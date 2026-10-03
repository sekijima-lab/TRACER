"""Offline one-time conversion of the fixed, author-published checkpoint."""
import argparse,hashlib,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--reference-repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.reference_repo.resolve()))
known={'ckpt_conditional.pth': '4ceaee8c8ee2249e4573e382a9532405501001ba16b027dcdfeea66f08e0ad9c', 'ckpt_unconditional.pth': 'da3623c02cc0c09c6c0d41b05380c8b43b21711111c26c070628dcc3aa6fed39'}
assert a.source.name in known and hashlib.sha256(a.source.read_bytes()).hexdigest()==known[a.source.name], 'Source is not the verified author checkpoint'
import torch
ckpt=torch.load(a.source,map_location='cpu',weights_only=False);state=ckpt['model_state_dict'];assert all(isinstance(v,torch.Tensor) for v in state.values());torch.save({'model_state_dict':state},a.output);print('CHECKPOINT_CONVERTED',a.output,{k:tuple(v.shape) for k,v in state.items() if k in ('embedding.weight','out.weight')},flush=True)

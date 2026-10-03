import argparse,sys,json,os,pickle
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--legacy',action='store_true');a=p.parse_args();a.repo=a.repo.resolve();a.checkpoint=a.checkpoint.resolve();a.out=a.out.resolve();sys.path.insert(0,str(a.repo));os.chdir(a.repo)
import numpy as np,torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*');torch.set_num_threads(1)
from scripts.preprocess import make_counter,make_transforms
from scripts.mcts import ParseSelectMCTS
from Model.Transformer.model import Transformer
from Model.GCN.network import MolecularGCN
from Utils.reward import QSAR_Reward
root=a.repo/'data/USPTO';d=make_counter(*[str(root/name) for name in ('src_train.txt','tgt_train.txt','src_valid.txt','tgt_valid.txt')]);s,t,v=make_transforms(d,make_vocab=True)
m=Transformer(d_model=512,nhead=8,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,dropout=.1,vocab=v,device='cpu');m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=not a.legacy)['model_state_dict']);m.eval();g=MolecularGCN(256,1,3,.1);g.load_state_dict(torch.load(a.repo/'ckpts/GCN/GCN.pth',map_location='cpu',weights_only=True));g.eval()
if a.legacy:
 with (a.repo/'Model/QSAR/qsar_DRD2_optimized.pkl').open('rb') as f:q=pickle.load(f)
else:
 from tracer_runtime.forest import NumericForest
 q=NumericForest(a.repo/'Model/QSAR/qsar_DRD2_optimized.npz')
r=QSAR_Reward(q);templates=json.loads((a.repo/'data/label_template.json').read_text());beam_templates=(a.repo/'data/beamsearch_template_list.txt').read_text().splitlines();inputs=(a.repo/'data/input/init_smiles_drd2.txt').read_text().splitlines()[:2];results=[]
for seed in [1729,19,73]:
 for smi in inputs:
  np.random.seed(seed);torch.manual_seed(seed)
  search=ParseSelectMCTS(smi,m,g,v,r,max_depth=2,c=1/np.sqrt(2),r_dict=templates,src_transforms=s,beam_width=3,nbest=2,inf_max_len=80,beam_templates=beam_templates,rollout_depth=1,device='cpu',GCN_device='cpu',exp_num_sampling=3,roll_num_sampling=2)
  search.search(2)
  result={'seed':seed,'input':smi,'step':search.step,'valid':search.n_valid,'invalid':search.n_invalid,'templates':search.gen_templates,'best_score':float(search.max_score),'molecules':[{'key':list(key),'score':float(value)} for key,value in search.valid_smiles.items()]};results.append(result);print('MCTS_CASE_DONE',seed,smi,len(result['molecules']),flush=True)
a.out.write_text(json.dumps(results,indent=2)+'\n');print('MCTS_DONE',flush=True)

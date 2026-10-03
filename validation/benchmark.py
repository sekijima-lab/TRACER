import argparse,sys,json,pickle
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--legacy',action='store_true');p.add_argument('--seed',type=int,default=1729);p.add_argument('--loss-kind',choices=['nll','ce'],default='nll');a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));a.out.mkdir(parents=True,exist_ok=True)
import numpy as np,torch,pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem,QED
RDLogger.DisableLog('rdApp.*');torch.set_num_threads(1)
from Model.GCN.mol2graph import mol2vec
from Model.GCN.network import MolecularGCN
from torch_geometric.data import Batch
from Utils.utils import smi_tokenizer
arrays={};results={}
smiles=[]
for name in ('AKT1','CXCR4','DRD2'):
 frame=pd.read_csv(a.repo/'data/QSAR'/name/(name.lower()+'_test.csv'))
 column=next((c for c in frame.columns if 'smile' in c.lower() or c.lower()=='canonical'),None)
 if column is None:
  frame=pd.read_csv(a.repo/'data/QSAR'/name/(name.lower()+'_test.csv'),header=None);column=1 if len(frame.columns)>1 else 0
 smiles.extend(frame[column].head(256).tolist())
smiles=list(dict.fromkeys(smiles));mols=[Chem.MolFromSmiles(s) for s in smiles];valid=[i for i,m in enumerate(mols) if m is not None];smiles=[smiles[i] for i in valid];mols=[mols[i] for i in valid]
fp=np.array([AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=2048) for m in mols]);arrays['fingerprints']=fp;arrays['qed']=np.array([QED.qed(m) for m in mols]);results['smiles']=smiles
for name in ('AKT1','CXCR4','DRD2'):
 if a.legacy:
  with (a.repo/'Model/QSAR'/('qsar_'+name+'_optimized.pkl')).open('rb') as f:model=pickle.load(f)
 else:
  from tracer_runtime.forest import NumericForest
  model=NumericForest(a.repo/'Model/QSAR'/('qsar_'+name+'_optimized.npz'))
 arrays['qsar_'+name]=model.predict_proba(pd.DataFrame(fp,columns=['bit_'+str(i) for i in range(2048)]))
model=MolecularGCN(256,1,3,.1);state=torch.load(a.repo/'ckpts/GCN/GCN.pth',map_location='cpu',weights_only=True);model.load_state_dict(state);model.eval()
graphs=[mol2vec(m) for m in mols[:32]];batch=Batch.from_data_list(graphs)
for k in ('x','edge_index','edge_attr','batch'): arrays['graph_'+k]=getattr(batch,k).numpy()
with torch.no_grad(): arrays['gcn_logits']=model(batch.x,batch.edge_index,batch.batch).numpy()
arrays['gcn_top10']=np.argsort(-arrays['gcn_logits'],axis=1)[:,:10]
model.train();torch.manual_seed(a.seed);optimizer=torch.optim.Adam(model.parameters(),lr=.0004);output=model(batch.x,batch.edge_index,batch.batch);
labels=torch.arange(32)%1000
if a.loss_kind=='ce':
 if a.legacy: loss=torch.nn.functional.cross_entropy(output,labels)
 else:
  from tracer_runtime.softmax import cross_entropy
  loss=cross_entropy(output,labels,model.old_compatible)
else: loss=torch.nn.functional.nll_loss(output,labels)
loss.backward();arrays['gcn_loss']=loss.detach().numpy();arrays['gcn_gradient']=torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy();optimizer.step();arrays['gcn_weights']=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy()
tokens=[smi_tokenizer(s) for s in smiles];results['tokenized_smiles']=tokens;results['packages']={k:__import__(k).__version__ for k in ['torch','torch_geometric','numpy','pandas','sklearn','rdkit']}
np.savez_compressed(a.out/'numeric.npz',**arrays);(a.out/'metadata.json').write_text(json.dumps(results,indent=2)+'\n');print('BENCHMARK_DONE',len(smiles),flush=True)

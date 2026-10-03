import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--legacy',action='store_true');a=p.parse_args();a.repo=a.repo.resolve();a.out=a.out.resolve();a.checkpoint=a.checkpoint.resolve();sys.path.insert(0,str(a.repo));__import__('os').chdir(a.repo);a.out.mkdir(parents=True,exist_ok=True)
import numpy as np,torch
from Model.Transformer.model import Transformer,TransformerLR
from scripts.preprocess import make_counter,make_transforms
from scripts.beam_search import greedy_translate,beam_decode
torch.set_num_threads(1);root=a.repo/'data/USPTO';d=make_counter(*[str(root/name) for name in ('src_train.txt','tgt_train.txt','src_valid.txt','tgt_valid.txt')]);s,t,v=make_transforms(d,make_vocab=True)
ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=not a.legacy);state=ckpt['model_state_dict'];vsize=state['embedding.weight'].shape[0]
if len(v)!=vsize:raise ValueError(('Vocabulary mismatch',len(v),vsize))
model=Transformer(d_model=512,nhead=8,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,dropout=0.0,vocab=v,device='cpu');model.load_state_dict(state);model.eval()
# Fixed real validation reactions, padded to each batch's maximum length.
src_tokens=d['datasets'][2][:8];tgt_tokens=d['datasets'][3][:8];src=s(src_tokens).T;tgt=t(tgt_tokens).T
src=src[:max(len(x) for x in src_tokens)];tgt=tgt[:max(len(x)+2 for x in tgt_tokens)];mask=torch.nn.Transformer.generate_square_subsequent_mask(len(tgt)-1)
r={'src':src.numpy(),'tgt':tgt.numpy()}
with torch.no_grad():r['logits']=model(src,tgt[:-1],tgt_mask=mask,src_pad_mask=True,tgt_pad_mask=True,memory_pad_mask=True).numpy()
r['argmax']=r['logits'].argmax(-1)
# Full trained-model gradient/Adam step; dropout disabled to isolate deterministic math.
model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.998));scheduler=TransformerLR(optimizer,warmup_epochs=8000);output=model(src,tgt[:-1],tgt_mask=mask,src_pad_mask=True,tgt_pad_mask=True,memory_pad_mask=True);loss=torch.nn.functional.cross_entropy(output.reshape(-1,len(v)),tgt[1:].reshape(-1),ignore_index=v['<pad>'],reduction='sum')/8;loss.backward();r['loss']=loss.detach().numpy();r['gradient']=torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();scheduler.step();r['weights']=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy();r['lr']=np.array(scheduler.get_last_lr())
# Generation uses the original learned weights, not the one-step updated weights.
model.load_state_dict(state);model.eval();generated=[]
for tokens in src_tokens[:3]:
 x=torch.tensor(v(tokens));greedy=greedy_translate(v=v,model=model,input_tokens=x.unsqueeze(0),device='cpu',inf_max_len=80)
 beam=beam_decode(v=v,model=model,input_tokens=x,template_idx=None,device='cpu',inf_max_len=80,beam_width=3,nbest=2,Temp=1,beam_templates=[])
 generated.append({'input':tokens,'greedy':greedy,'beam':beam})
np.savez_compressed(a.out/'numeric.npz',**r);(a.out/'generation.json').write_text(json.dumps(generated,indent=2)+'\n');print('TRANSFORMER_BENCHMARK_DONE',len(v),float(loss),flush=True)

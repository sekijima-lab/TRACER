import os,sys,subprocess,json,torch
torch.set_num_threads(1)
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.repo.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);fixture=root/'validation/fixtures';fixture.mkdir(parents=True,exist_ok=True)
for name in ['src_train.txt','tgt_train.txt','src_valid.txt','tgt_valid.txt']:
 source=(root/'data/USPTO'/name).read_text().splitlines();(fixture/name).write_text('\n'.join(source[:16])+'\n')
base={'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
for mode in ['1','0']:
 env={**os.environ,**base,'TRACER_OLD_COMPATIBLE':mode,'TRACER_DEVICE':'cpu'}
 train=[sys.executable,'scripts/transformer_train.py','model.dim_model=32','model.nhead=4','model.num_encoder_layers=1','model.num_decoder_layers=1','model.dim_ff=64','model.dropout=0','train.batch_size=16','train.step_num=1','train.log_interval=2','train.val_interval=2','train.save_interval=2']+[f'train.{name}=/validation/fixtures/{name}.txt' for name in ['src_train','tgt_train','src_valid','tgt_valid']]
 before=set((root/'ckpts').glob('checkpoints_*'))
 with (out/('transformer-'+mode+'.log')).open('w') as f:subprocess.run(train,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 created=set((root/'ckpts').glob('checkpoints_*'))-before
 assert len(created)==1
 run=created.pop();checkpoint=next(run.glob('ckpt_*.pth'));ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert ckpt['runtime']['old_compatible']==(mode=='1') and ckpt['optimizer_state_dict']['state']
 assert json.loads((run/'runtime.json').read_text())['old_compatible']==(mode=='1')
 # Restricted saved tensor checkpoint and optimizer state can both be restored.
 sys.path.insert(0,str(root))
 from scripts.preprocess import make_counter,make_transforms
 from Model.Transformer.model import Transformer
 dictionary=make_counter(*[str(fixture/name) for name in ['src_train.txt','tgt_train.txt','src_valid.txt','tgt_valid.txt']]);src_transform,tgt_transform,vocab=make_transforms(dictionary,make_vocab=True)
 model=Transformer(d_model=32,nhead=4,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=64,dropout=0,vocab=vocab,device='cpu',old_compatible=mode=='1');model.load_state_dict(ckpt['model_state_dict'])
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.998));optimizer.load_state_dict(ckpt['optimizer_state_dict'])
 src=src_transform(dictionary['datasets'][0][:4]).T;tgt=tgt_transform(dictionary['datasets'][1][:4]).T;mask=torch.nn.Transformer.generate_square_subsequent_mask(len(tgt)-1)
 model.train();output=model(src,tgt[:-1],tgt_mask=mask,src_pad_mask=True,tgt_pad_mask=True,memory_pad_mask=True);loss=torch.nn.functional.cross_entropy(output.reshape(-1,len(vocab)),tgt[1:].reshape(-1),ignore_index=vocab['<pad>']);loss.backward();optimizer.step()
 assert all(torch.isfinite(p).all() for p in model.parameters())
 gcn=[sys.executable,'scripts/gcn_train.py','GCN_train.dim=32','GCN_train.n_conv_hidden=1','GCN_train.n_mlp_hidden=1','GCN_train.batch_size=16','GCN_train.epochs=1','GCN_train.train=/validation/fixtures/src_train.txt','GCN_train.valid=/validation/fixtures/src_valid.txt',f'GCN_train.save_path=/validation/cli-gcn-{mode}']
 with (out/('gcn-'+mode+'.log')).open('w') as f:subprocess.run(gcn,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 gcnroot=root/('validation/cli-gcn-'+mode);runs=list(gcnroot.glob('checkpoints_*'));run=max(runs,key=lambda p:p.stat().st_mtime);state=torch.load(run/'ckpt.pth',map_location='cpu',weights_only=True);assert state and all(isinstance(v,torch.Tensor) for v in state.values());assert json.loads((run/'runtime.json').read_text())['old_compatible']==(mode=='1')
 print('CLI_TRAIN_PASS',mode,flush=True)
(out/'result.json').write_text(json.dumps({'both_modes_transformer_training':True,'both_modes_gcn_training':True,'limited_batches_and_steps':True,'restricted_reload_and_transformer_optimizer_resume':True,'runtime_mode_records':True},indent=2)+'\n')

"""Compare full benchmark outputs or the committed compact historical references."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--gcn',type=Path,required=True);p.add_argument('--transformer',type=Path,required=True);p.add_argument('--legacy-gcn',type=Path);p.add_argument('--legacy-transformer',type=Path);a=p.parse_args();base=Path(__file__).resolve().parent;criteria=json.loads((base/'criteria.json').read_text());results={}
for kind,folder,reference in [('gcn',a.gcn,a.legacy_gcn or base/'legacy-gcn-numeric.npz'),('transformer',a.transformer,a.legacy_transformer or base/'legacy-transformer-numeric.npz')]:
    with np.load(reference,allow_pickle=False) as old,np.load(folder/'numeric.npz',allow_pickle=False) as new:
        for key in old.files:
            if key.endswith('_indices'):continue
            x,y=old[key],new[key]
            if key+'_indices' in old.files:y=y.reshape(-1)[old[key+'_indices']]
            assert x.shape==y.shape,key
            assert np.isfinite(x).all() and np.isfinite(y).all(),key
            difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
            if key.startswith('graph_') or key in ('fingerprints','gcn_top10','src','tgt','argmax','lr'):limit=0
            elif key.startswith('qsar_') or key=='qed':limit=criteria['qsar_probabilities_max_abs']
            elif 'logits' in key:limit=criteria[kind+'_logits_max_abs']
            elif 'loss' in key:limit=criteria['training_loss_max_abs']
            elif 'gradient' in key:limit=criteria['training_gradient_max_abs']
            elif 'weights' in key:limit=criteria['one_optimizer_step_weights_max_abs']
            else:raise ValueError('Unrecognized benchmark key '+key)
            assert difference<=limit,(kind,key,difference,limit)
            results[kind+':'+key]=difference
old_generation=json.loads((base/'legacy-generation.json').read_text());new_generation=json.loads((a.transformer/'generation.json').read_text());assert old_generation==new_generation,'Generation differs'
print(json.dumps({'verdict':'PASS','scope':'Full arrays when full baselines supplied; otherwise compact gradient/weight samples','max_abs':results},indent=2))

"""One-time trusted repository pickle conversion in the isolated reference environment."""
import pickle, json, hashlib, argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();root=args.source;out=args.out;out.mkdir(parents=True,exist_ok=True);records=[]
known={'qsar_AKT1_optimized.pkl': '06b15206f5a79076646130dd07bea5bc9111b278a27eb55c978d23f57855c56f', 'qsar_CXCR4_optimized.pkl': '29e8e71fb9791bcdc99a339357ac8e5b2a3660e5f1743c011b55cea61d202e4a', 'qsar_DRD2_optimized.pkl': '2db876d2c1d2831594021419fdf8b037c624e68126b3cc240116f99b3c1609aa'}
for path in sorted(root.glob('*.pkl')):
 assert path.name in known and hashlib.sha256(path.read_bytes()).hexdigest()==known[path.name], 'Unknown QSAR pickle; expected fixed trusted source'
 with path.open('rb') as f: model=pickle.load(f)
 offsets=[0];left=[];right=[];feature=[];threshold=[];prob=[]
 for estimator in model.estimators_:
  t=estimator.tree_;left.append(t.children_left);right.append(t.children_right);feature.append(t.feature);threshold.append(t.threshold)
  v=t.value[:,0,:];prob.append(v/v.sum(1,keepdims=True));offsets.append(offsets[-1]+t.node_count)
 dest=out/(path.stem+'.npz')
 np.savez_compressed(dest, offsets=np.array(offsets,dtype=np.int64),left=np.concatenate(left),right=np.concatenate(right),feature=np.concatenate(feature),threshold=np.concatenate(threshold),probability=np.concatenate(prob),classes=model.classes_,n_features=np.array(model.n_features_in_))
 records.append({'source':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'converted':dest.name,'converted_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'trees':len(model.estimators_),'features':model.n_features_in_})
(out/'qsar-conversion.json').write_text(json.dumps(records,indent=2)+'\n')

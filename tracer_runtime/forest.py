"""Inference-only numerical representation of the bundled binary random forests."""
from pathlib import Path
import numpy as np

class NumericForest:
    def __init__(self, path):
        with np.load(Path(path),allow_pickle=False) as data:
            self.data={key:data[key] for key in data.files}
        d=self.data
        self.classes_=d['classes']
        self.n_features_in_=int(d['n_features'])
        if not np.array_equal(self.classes_,[0,1]): raise ValueError('Expected binary classes [0,1]')
        offsets=d['offsets'];total=int(offsets[-1])
        if offsets.ndim!=1 or len(offsets)<2 or offsets[0]!=0 or not np.all(np.diff(offsets)>0): raise ValueError('Invalid tree offsets')
        for key in ('left','right','feature','threshold'):
            if d[key].shape!=(total,): raise ValueError('Invalid tree array')
        if d['probability'].shape!=(total,2) or not np.isfinite(d['probability']).all(): raise ValueError('Invalid probabilities')
        for start,end in zip(offsets[:-1],offsets[1:]):
            size=end-start;left=d['left'][start:end];right=d['right'][start:end];features=d['feature'][start:end]
            branch=left!=-1
            if np.any((right==-1)!=~branch) or np.any(left[branch]<0) or np.any(right[branch]<0) or np.any(left[branch]>=size) or np.any(right[branch]>=size): raise ValueError('Invalid tree children')
            if np.any(features[branch]<0) or np.any(features[branch]>=self.n_features_in_): raise ValueError('Invalid feature indices')
    def predict_proba(self, X):
        x=np.asarray(X,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not np.isfinite(x).all(): raise ValueError('Expected finite fingerprint matrix')
        d=self.data;result=np.zeros((len(x),2),dtype=np.float64)
        for start,end in zip(d['offsets'][:-1],d['offsets'][1:]):
            nodes=np.zeros(len(x),dtype=np.int64)
            active=np.arange(len(x));steps=0
            while len(active):
                at=start+nodes[active];branch=d['left'][at]!=-1
                finished=active[~branch];result[finished]+=d['probability'][start+nodes[finished]]
                active=active[branch]
                at=start+nodes[active]
                nodes[active]=np.where(x[active,d['feature'][at]]<=d['threshold'][at],d['left'][at],d['right'][at])
                steps+=1
                if steps>end-start: raise ValueError('Cyclic tree')
        return result/(len(d['offsets'])-1)

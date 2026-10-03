"""Ordered vocabulary and padding used by TRACER, without torchtext native code."""
import json
from pathlib import Path
import torch

class Vocabulary:
    def __init__(self, tokens, default_index=None):
        self.tokens = list(tokens)
        if len(set(self.tokens)) != len(self.tokens) or not all(isinstance(t,str) for t in self.tokens):
            raise ValueError('Vocabulary requires unique string tokens')
        self.indices = {t:i for i,t in enumerate(self.tokens)}
        self.set_default_index(default_index)
    def __len__(self): return len(self.tokens)
    def __getitem__(self, token):
        if token in self.indices: return self.indices[token]
        if self.default_index is None: raise KeyError(token)
        return self.default_index
    def __call__(self, tokens): return [self[t] for t in tokens]
    def get_itos(self): return list(self.tokens)
    def get_stoi(self): return dict(self.indices)
    def lookup_tokens(self, indices):
        if any(i < 0 or i >= len(self) for i in indices): raise IndexError('Token index outside vocabulary')
        return [self.tokens[i] for i in indices]
    def set_default_index(self, index):
        if index is not None and (not isinstance(index,int) or not 0 <= index < len(self)): raise ValueError('Invalid default index')
        self.default_index = index
    def get_default_index(self): return self.default_index


def vocab(counter, min_freq=1, specials=()):
    # torchtext.vocab.vocab preserves insertion order and places specials first.
    return Vocabulary(list(specials) + [t for t,n in counter.items() if n >= min_freq and t not in specials])

def save_vocab(v, path):
    Path(path).write_text(json.dumps({'tokens':v.get_itos(),'default_index':v.get_default_index()},ensure_ascii=False)+'\n')

def load_vocab(path):
    data=json.loads(Path(path).read_text())
    return Vocabulary(data['tokens'],data['default_index'])

class TokenTransform:
    def __init__(self, v, length, target=False):
        self.v, self.length, self.target = v, length, target
    def __call__(self, sequences):
        if sequences and isinstance(sequences[0], str):
            row=self.v(sequences)
            if self.target: row=[self.v['<bos>']]+row+[self.v['<eos>']]
            result=torch.tensor(row,dtype=torch.long)
        else:
            rows=[self.v(seq) for seq in sequences]
            if self.target: rows = [([self.v['<bos>']] + row + [self.v['<eos>']]) for row in rows]
            result=torch.nn.utils.rnn.pad_sequence([torch.tensor(row,dtype=torch.long) for row in rows],batch_first=True,padding_value=self.v['<pad>'])
        if result.shape[-1]<self.length:
            result=torch.nn.functional.pad(result,(0,self.length-result.shape[-1]),value=self.v['<pad>'])
        return result

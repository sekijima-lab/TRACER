import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.init import xavier_uniform_
import torchtext.vocab.vocab as Vocab



class PositionalEncoding(nn.Module):
# P(pos, 2d) = sin(pos/10000**(2d/D)), where d=index of token, D=d_model
    def __init__(self, d_model: int, pad_idx: int=1, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.pad_idx = pad_idx

        position = torch.arange(max_len).unsqueeze(1) # shape: (max_len, 1) の列ベクトル
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        # torch.arange(start, stop, step) -> shape: (d_model/2,) 1次元ベクトル
        pe = torch.zeros(max_len, 1, self.d_model) # (seq_length, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.tensor, pad_mask=None) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if pad_mask is not None:
            mask = pad_mask.permute(1, 0).unsqueeze(-1).repeat(1, 1, self.d_model) # paddingの位置をTrue
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)] # max_lenが5000とかでも入力のseq_lenまでを切り取って足してる
        
        if pad_mask is not None:
            x = torch.where(mask == True, 0, x) # mask=Trueの位置を0に置換, それ以外はいじらない
        return self.dropout(x)


# Learning Rate Scheduler
class TransformerLR(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, warmup_epochs=8000, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_epochs = warmup_epochs
        self.normalize = self.warmup_epochs**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_epochs**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


# Transformer model
class Transformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4, num_decoder_layers: int =4,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 vocab: Vocab = None, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if vocab == None:
            raise RuntimeError("set vocab: torch.vocab.vocab")
        
        # INFO
        self.model_type = "Transformer"
        self.vocab = vocab
        num_tokens = vocab.__len__()
        
        self.positional_encoder = PositionalEncoding(d_model=d_model,
                                                     pad_idx=self.vocab['<pad>'],
                                                     dropout=dropout,
                                                     max_len=5000
                                                     )
        self.embedding = nn.Embedding(num_tokens, d_model, padding_idx=self.vocab['<pad>'])

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.out = nn.Linear(d_model, num_tokens)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        

    # Transformer blocks - Out size = (seq_length, batch_size, num_tokens)
    # input src, tgt must be (seq_length, batch_size)
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_pad_mask: bool = False, tgt_pad_mask: bool = False,
                memory_pad_mask: bool = False) -> Tensor:
        
        if src_pad_mask is True:
            src_pad_mask = (src == self.vocab['<pad>']).permute(1, 0)
        else:
            src_pad_mask = None
            
        if tgt_pad_mask is True:
            tgt_pad_mask = (tgt == self.vocab['<pad>']).permute(1, 0)
        else:
            tgt_pad_mask = None
        
        if memory_pad_mask is True:
            memory_pad_mask = (src == self.vocab['<pad>']).permute(1, 0)
        else:
            memory_pad_mask = None
        
        # Embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoder(src, src_pad_mask)
        tgt = self.positional_encoder(tgt, tgt_pad_mask)
        
        # Transformer layer
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        
        memory = self.encoder(src=src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        transformer_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_pad_mask)
        out = self.out(transformer_out)
        
        return out
    
    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None,
               src_pad_mask: bool = False) -> Tensor:
        
        if src_pad_mask is True:
            src_pad_mask = (src == self.vocab['<pad>']).permute(1, 0)
        else:
            src_pad_mask = None
        
        # Embedding + PE
        src = self.embedding(src)
        src = self.positional_encoder(src, src_pad_mask)

        # Transformer Encoder
        memory = self.encoder(src=src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        
        return memory, src_pad_mask
    
    def decode(self, memory: Tensor, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
               memory_mask: Optional[Tensor] = None, tgt_pad_mask: bool = False,
               memory_pad_mask: Optional[Tensor] = None) -> Tensor:
        
        if tgt_pad_mask is True:
            tgt_pad_mask = (tgt == self.vocab['<pad>']).permute(1, 0)
        else:
            tgt_pad_mask = None
            
        tgt = self.embedding(tgt)
        tgt = self.positional_encoder(tgt, tgt_pad_mask)
        transformer_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_pad_mask)
        out = self.out(transformer_out)
        
        return out
    
    def embed(self, src):
        src_embed = self.embedding(src)
        return src_embed

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
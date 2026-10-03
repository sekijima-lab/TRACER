"""Historical CPU attention scaling and scalar-libm softmax; current tensor operators."""
import math
import torch
from torch.nn import functional as F
from .softmax import softmax

class CompatibleMultiheadAttention(torch.nn.MultiheadAttention):
    def forward(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,average_attn_weights=True,is_causal=False):
        if query.device.type!='cpu' or query.dtype!=torch.float32 or query.ndim!=3 or self.batch_first or not self._qkv_same_embed_dim or self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise ValueError('Old-compatible attention requires sequence-first 3-D CPU float32 and standard equal-dimension Q/K/V')
        length,batch,dimension=query.shape;source_length=key.shape[0];heads=self.num_heads;head_dim=dimension//heads
        if key.shape!=value.shape or key.shape[1:]!=query.shape[1:]:raise ValueError('Invalid attention Q/K/V shapes')
        # Same packed projection order used by torch 2.0.1 multi_head_attention_forward.
        q,k,v=F._in_projection_packed(query,key,value,self.in_proj_weight,self.in_proj_bias)
        q=q.view(length,batch*heads,head_dim).transpose(0,1).reshape(batch,heads,length,head_dim)
        k=k.view(source_length,batch*heads,head_dim).transpose(0,1).reshape(batch,heads,source_length,head_dim)
        v=v.view(source_length,batch*heads,head_dim).transpose(0,1).reshape(batch,heads,source_length,head_dim)
        factor=math.sqrt(math.sqrt(head_dim))
        scores=torch.matmul(q/factor,k.transpose(-2,-1)/factor)
        if attn_mask is not None:
            if attn_mask.shape==(length,source_length):mask=attn_mask.reshape(1,1,length,source_length)
            elif attn_mask.shape==(batch*heads,length,source_length):mask=attn_mask.reshape(batch,heads,length,source_length)
            else:raise ValueError('Invalid attention mask shape')
            scores=scores.masked_fill(mask,float('-inf')) if mask.dtype==torch.bool else scores+mask
        elif is_causal:
            mask=torch.ones(length,source_length,dtype=torch.bool).triu(1);scores=scores.masked_fill(mask,float('-inf'))
        if key_padding_mask is not None:
            if key_padding_mask.shape!=(batch,source_length):raise ValueError('Invalid key padding mask shape')
            mask=key_padding_mask.reshape(batch,1,1,source_length)
            scores=scores.masked_fill(mask,float('-inf')) if mask.dtype==torch.bool else scores+mask
        probability=softmax(scores)
        if self.training and self.dropout:probability=F.dropout(probability,self.dropout,True)
        attended=torch.matmul(probability,v).permute(2,0,1,3).contiguous().view(length*batch,dimension)
        output=F.linear(attended,self.out_proj.weight,self.out_proj.bias).view(length,batch,dimension)
        weights=probability.mean(1) if average_attn_weights else probability
        return output,weights if need_weights else None

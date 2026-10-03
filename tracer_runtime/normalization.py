"""Old CPU float32 LayerNorm statistics/affine order; current native first gradients."""
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from . import _legacy_math

class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,eps):
        if x.device.type!='cpu' or x.dtype!=torch.float32:raise ValueError('Old-compatible LayerNorm requires CPU float32')
        source=np.ascontiguousarray(x.detach().numpy());shape=(*x.shape[:-1],1);mean=np.empty(shape,np.float32);inv=np.empty(shape,np.float32)
        _legacy_math.moments(source,mean,inv,x.shape[-1],eps)
        mean,inv=torch.from_numpy(mean),torch.from_numpy(inv)
        ctx.save_for_backward(x,weight,bias,mean,inv)
        return (x*inv+(-inv*mean))*weight+bias
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,weight,bias,mean,inv=ctx.saved_tensors
        dx,dw,db=torch.ops.aten.native_layer_norm_backward(grad.contiguous(),x,[x.shape[-1]],mean,inv,weight,bias,[True,True,True])
        return dx,dw,db,None

class CompatibleLayerNorm(torch.nn.LayerNorm):
    def forward(self,x):
        if len(self.normalized_shape)!=1 or self.weight is None or self.bias is None:raise ValueError('Expected one-dimensional affine LayerNorm')
        return _LayerNorm.apply(x,self.weight,self.bias,self.eps)

"""Historical scalar-libm log-softmax for the GCN classifier."""
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from . import _legacy_math

class _LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        if x.device.type!='cpu' or x.dtype!=torch.float32:raise ValueError('Old-compatible log-softmax requires CPU float32')
        source=np.ascontiguousarray(x.detach().numpy());out=np.empty_like(source);_legacy_math.logsoftmax(source,out,x.shape[-1]);output=torch.from_numpy(out);ctx.save_for_backward(output);return output
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        output,=ctx.saved_tensors;g=np.ascontiguousarray(grad.detach().numpy());out=np.empty_like(g);_legacy_math.logsoftmax_backward(g,output.numpy(),out,g.shape[-1]);return torch.from_numpy(out)

def log_softmax(x,old_compatible):
    return _LogSoftmax.apply(x) if old_compatible else torch.nn.functional.log_softmax(x,dim=-1)

class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        if x.device.type!='cpu' or x.dtype!=torch.float32:raise ValueError('Old-compatible softmax requires CPU float32')
        source=np.ascontiguousarray(x.detach().numpy());out=np.empty_like(source);_legacy_math.softmax(source,out,x.shape[-1]);output=torch.from_numpy(out);ctx.save_for_backward(output);return output
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        output,=ctx.saved_tensors;g=np.ascontiguousarray(grad.detach().numpy());out=np.empty_like(g);_legacy_math.softmax_backward(g,output.numpy(),out,g.shape[-1]);return torch.from_numpy(out)

def softmax(x):
    return _Softmax.apply(x)

def cross_entropy(logits,target,old_compatible):
    if not old_compatible:
        return torch.nn.functional.cross_entropy(logits,target)
    return torch.nn.functional.nll_loss(log_softmax(logits,True),target)

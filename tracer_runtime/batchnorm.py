"""Historical CPU BatchNorm affine order, using current native statistics/gradients."""
import torch
from torch.autograd.function import once_differentiable

class _TrainingBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,running_mean,running_var,momentum,eps):
        _,mean,invstd=torch.ops.aten.native_batch_norm(x,weight,bias,running_mean,running_var,True,momentum,eps)
        ctx.save_for_backward(x,weight,mean,invstd)
        ctx.eps=eps
        alpha=invstd*weight
        beta=bias-mean*alpha
        return x*alpha+beta
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,weight,mean,invstd=ctx.saved_tensors
        dx,dw,db=torch.ops.aten.native_batch_norm_backward(grad.contiguous(),x,weight,None,None,mean,invstd,True,ctx.eps,[True,True,True])
        return dx,dw,db,None,None,None,None

class CompatibleBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self,x):
        if x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=2 or not self.affine or not self.track_running_stats:
            raise ValueError('Old-compatible BatchNorm requires 2-D CPU float32 and affine/running statistics')
        if self.training:
            if len(x)<2:raise ValueError('Training BatchNorm requires at least two rows')
            self.num_batches_tracked.add_(1)
            momentum=self.momentum if self.momentum is not None else 1/float(self.num_batches_tracked)
            return _TrainingBatchNorm.apply(x,self.weight,self.bias,self.running_mean,self.running_var,momentum,self.eps)
        invstd=1/torch.sqrt(self.running_var+self.eps)
        alpha=invstd*self.weight
        beta=self.bias-self.running_mean*alpha
        return x*alpha+beta

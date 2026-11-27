"""Optional native CUDA FP32 pointwise fusion for the frozen research graph."""
import ctypes as C
import torch


class FusedPointwise:
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        self.lib.nr_unary.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_int,C.c_void_p]
        self.lib.nr_unary.restype=C.c_int
        self.lib.nr_residual.argtypes=[C.c_void_p]*4+[C.c_int64,C.c_int64,C.c_void_p]
        self.lib.nr_residual.restype=C.c_int
        self.calls={"silu":0,"exp":0,"residual":0}

    @staticmethod
    def supported(x):
        return x.is_cuda and x.dtype==torch.float32 and x.is_contiguous()

    @staticmethod
    def check(code):
        if code: raise RuntimeError(f"native pointwise CUDA error {code}")

    def unary(self,x,op):
        if not self.supported(x): raise ValueError("expected contiguous CUDA FP32")
        with torch.cuda.device(x.device):
            y=torch.empty_like(x)
            self.check(self.lib.nr_unary(x.data_ptr(),y.data_ptr(),x.numel(),op,
                                        torch.cuda.current_stream(x.device).cuda_stream))
        return y

    def silu(self,x):
        self.calls["silu"]+=1
        return self.unary(x,0)

    def exp(self,x,global_mode=False):
        self.calls["exp"]+=1
        return self.unary(x,2 if global_mode else 1)

    def supports_residual(self,x,skip,scale):
        return (all(self.supported(t) and t.device==x.device for t in (x,skip,scale))
                and x.shape==skip.shape and x.ndim>0
                and scale.ndim<=x.ndim
                and (scale.numel()==1 or
                     (scale.numel()==x.shape[-1] and scale.ndim<=x.ndim
                      and scale.shape[-1]==x.shape[-1]
                      and all(s==1 for s in scale.shape[:-1]))))

    def residual(self,x,skip,scale):
        if not self.supports_residual(x,skip,scale): raise ValueError("unsupported residual layout")
        self.calls["residual"]+=1
        with torch.cuda.device(x.device):
            y=torch.empty_like(x)
            self.check(self.lib.nr_residual(x.data_ptr(),skip.data_ptr(),scale.data_ptr(),
                        y.data_ptr(),x.numel(),scale.numel(),
                        torch.cuda.current_stream(x.device).cuda_stream))
        return y


def install(model,path):
    """Retains upstream behavior for unsupported layouts and dtypes."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("requires frozen inference model")
    f=FusedPointwise(path)
    n=model.numerics
    silu,exp,residual=n.silu,n.exp,n.residual
    n.silu=lambda x: f.silu(x) if f.supported(x) else silu(x)
    n.exp=lambda x,global_mode=False: f.exp(x,global_mode) if f.supported(x) else exp(x,global_mode)
    n.residual=lambda x,s,k: f.residual(x,s,k) if f.supports_residual(x,s,k) else residual(x,s,k)
    return f

"""Opt-in W8A8 linear probe for a frozen reconstructed model.

Per-output-channel weight scales; dynamic per-row activation scales.
Uses private Torch CUDA int8 GEMM; not a production backend.
Attention matmuls, reductions, sensitive small linear layers stay floating.
The original buffers are retained; cached int8 copies are not VRAM savings.
"""
import torch


def quantize_rows(x):
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("expected floating two-dimensional matrix")
    x=x.float()
    scale=x.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)/127.
    return (x/scale).round().clamp(-127,127).to(torch.int8),scale


class Int8Linear:
    def __init__(self,min_features=64):
        self.min_features=min_features
        self.cache={}
        self.quantized_calls=0
        self.fallback_calls=0

    def __call__(self,x,w):
        if (not x.is_cuda or x.device != w.device or w.ndim != 2
                or x.shape[-1]!=w.shape[-1] or min(w.shape)<self.min_features
                or w.shape[0]%16 or w.shape[1]%16
                or x.numel()//x.shape[-1]<32
                or (x.numel()//x.shape[-1])%8):
            self.fallback_calls+=1
            return None
        # The graph is frozen. Keep a reference to each immutable buffer/view
        # for the lifetime of the cache, including expert slices.
        key=(w.device,w.data_ptr(),tuple(w.shape),tuple(w.stride()))
        if key not in self.cache:
            q,scale=quantize_rows(w)
            self.cache[key]=(w,q.contiguous(),scale)
        _,qw,sw=self.cache[key]
        qx,sx=quantize_rows(x.reshape(-1,x.shape[-1]))
        raw=torch._int_mm(qx.contiguous(),qw.T)
        y=(raw.float()*sx)*sw.T
        self.quantized_calls+=1
        return y.reshape(*x.shape[:-1],w.shape[0])

    def stats(self):
        return dict(quantized_calls=self.quantized_calls,
                    fallback_calls=self.fallback_calls,
                    cached_weights=len(self.cache),
                    int8_cache_bytes=sum(q.numel()*q.element_size()+s.numel()*s.element_size()
                                         for _,q,s in self.cache.values()))


def install(model,min_features=64):
    """Wrap the shared numerics of a frozen inference graph, not training."""
    n=model.numerics
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("requires frozen eval model")
    original=n.linear
    engine=Int8Linear(min_features)
    def linear(x,weight,residual=None,scale=None,partition=0,*,f16_weights=False):
        # Respect the original floating frontend/head contract.
        raw=None if f16_weights else engine(x,weight)
        if raw is None:
            return original(x,weight,residual,scale,partition,f16_weights=f16_weights)
        raw=raw.to(n.working_dtype)
        return n.residual(raw,residual,scale) if residual is not None else raw
    n.linear=linear
    return engine

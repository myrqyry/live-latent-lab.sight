"""Persistent fixed-resolution RGB24 neural video worker.

Raw input: width*height*3 RGB24 bytes per frame. Default output is RGB24;
optional NRF1 output carries signed little-endian float32 RGB corrections.
Stdout is binary only; diagnostics go to stderr. EOF ends the stream.
One model and one CUDA stream per process. No implicit frame dropping.
Stateless reconstruction; fixed noise field, no motion/history reprojection.
"""
import argparse
from collections import deque
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from still import make_features
from input_conditioning import condition_input, transfer_correction, unit


def read_frame(stream, size):
    if size <= 0:
        raise ValueError("frame size must be positive")
    data = bytearray()
    while len(data) < size:
        part = stream.read(size-len(data))
        if not part:
            if data:
                raise EOFError(f"truncated RGB frame: {len(data)}/{size} bytes")
            return None
        data.extend(part)
    return bytes(data)


def process_frames(src, dst, size, render, limit=None, output_size=None):
    output_size = size if output_size is None else output_size
    count = 0
    while limit is None or count < limit:
        frame = read_frame(src, size)
        if frame is None:
            break
        result = render(frame)
        if len(result) != output_size:
            raise ValueError("renderer returned wrong frame size")
        written = dst.write(result)
        if written != output_size:
            raise IOError("short frame write")
        dst.flush()
        count += 1
    return count


class FeatureBuffer:
    """Reuse fixed noise, controls and reflected coordinates across frames."""
    def __init__(self, proxy, full_w, full_h, seed=320, structure=1.):
        self.features = make_features(proxy,full_w,full_h,seed=seed,structure=unit(structure))
        self.shape = proxy.shape
        self.device = proxy.device
        self.dtype = proxy.dtype
        def indices(n, extent):
            i = torch.arange(extent,device=proxy.device) % (2*n-2)
            return torch.minimum(i,2*n-2-i)
        self.rows = indices(proxy.shape[1], full_h)
        self.cols = indices(proxy.shape[2], full_w)

    def update(self, proxy):
        if (proxy.shape,proxy.device,proxy.dtype)!=(self.shape,self.device,self.dtype):
            raise ValueError("fixed RGB feature input contract changed")
        rgb = proxy[:,self.rows[:,None],self.cols[None,:],:]
        half = lambda x: x.half().float()
        centered = half(half(half(rgb)-.5)*.125)
        self.features[...,4:7].copy_(centered)
        self.features[...,7:10].copy_(centered)
        return self.features


def encode_residual(original, conditioned, head, intensity, mode):
    """Versioned little-endian float32 signed RGB correction, no alpha.

    Clipped mode keeps the old tone bounds without RGB24 rounding.
    Raw mode transfers the model's pre-clamp correction (not HDR).
    """
    if mode not in ("residual-f32","raw-residual-f32"):
        raise ValueError("unknown residual format")
    if not 0 <= intensity <= 1:
        raise ValueError("invalid intensity")
    if head.shape[:-1] != original.shape[:-1] or head.shape[-1] != 4 or conditioned.shape != original.shape:
        raise ValueError("invalid residual shapes")
    delta=head[...,:3].float()*(intensity/4.)
    if mode=="residual-f32":
        # Match the existing compose + transfer + final clamp contract.
        delta=((conditioned+head[...,:3]/4).clamp(0.,1.)-conditioned)*intensity
        delta=(original+delta).clamp(0.,1.)-original
    if not torch.isfinite(head).all().item() or not torch.isfinite(delta).all().item():
        raise RuntimeError("nonfinite neural residual")
    if (delta.abs()>16).any().item():
        raise RuntimeError("neural residual exceeds transport bounds")
    return b"NRF1"+delta.contiguous().cpu().numpy().astype("<f4",copy=False).tobytes()


class FrameRenderer:
    """Owns model, frame buffers, precision and optional capture."""
    @torch.inference_mode()
    def __init__(self, args):
        import numpy as np
        from dlssnr import Geometry, load_checkpoint
        self.np = np
        self.width, self.height = args.width,args.height
        self.intensity = args.intensity
        self.output_format = getattr(args,"output_format","rgb24")
        self.clarity = unit(args.input_clarity)
        self.shadows = unit(args.shadow_recovery)
        self.geometry = Geometry.from_valid(self.width,self.height)
        self.model = load_checkpoint(args.checkpoint,model="small31").training_model(
            device="cuda:0",precision="fp32",trainable=False,checkpoint_blocks=False).eval()
        self.fused = self.quantized = None
        if args.fused_so:
            from fused import install
            self.fused = install(self.model,args.fused_so.resolve())
        if args.precision=="w8a8":
            from quantized import install
            self.quantized = install(self.model,min_features=args.int8_min_features)
        self.proxy = torch.zeros((1,self.height,self.width,3),device="cuda",dtype=torch.float32)
        self.upload = torch.empty(self.proxy.shape,dtype=torch.uint8,device="cuda")
        self.feature_buffer = FeatureBuffer(self.proxy,self.geometry.full_width,
                                            self.geometry.full_height,args.seed,structure=args.structure)
        self.forward = lambda f: self.model(f,geometry=self.geometry,crop=True)
        if args.cuda_graph:
            from replay import FixedReplay
            self.forward = FixedReplay(self.forward,self.feature_buffer.features)
        else:
            for _ in range(3):
                self.forward(self.feature_buffer.features)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, frame):
        host = self.np.frombuffer(frame,dtype=self.np.uint8).copy().reshape(
            1,self.height,self.width,3)
        self.upload.copy_(torch.from_numpy(host))
        self.proxy.copy_(self.upload).div_(255.)
        conditioned = condition_input(self.proxy,self.clarity,self.shadows)
        head = self.forward(self.feature_buffer.update(conditioned))
        if self.output_format!="rgb24":
            return encode_residual(self.proxy,conditioned,head,self.intensity,self.output_format)
        out = transfer_correction(self.proxy,conditioned,head,self.intensity)
        if not torch.isfinite(head).all().item() or not torch.isfinite(out).all().item():
            raise RuntimeError("nonfinite neural frame")
        return out.mul(255).round().byte().cpu().numpy().tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--width",required=True,type=int)
    p.add_argument("--height",required=True,type=int)
    p.add_argument("--precision",choices=("fp32","w8a8"),default="fp32")
    p.add_argument("--int8-min-features",type=int,default=256)
    p.add_argument("--fused-so",type=Path)
    p.add_argument("--cuda-graph",action="store_true")
    p.add_argument("--intensity",type=float,default=1.)
    p.add_argument("--output-format",choices=("rgb24","residual-f32","raw-residual-f32"),default="rgb24")
    p.add_argument("--seed",type=int,default=320)
    p.add_argument("--structure",type=float,default=1.)
    p.add_argument("--input-clarity",type=float,default=0.)
    p.add_argument("--shadow-recovery",type=float,default=0.)
    p.add_argument("--limit",type=int)
    p.add_argument("--receipt",type=Path)
    args = p.parse_args()
    if min(args.width,args.height)<33 or not 0<=args.intensity<=1:
        p.error("dimensions must be >=33 and intensity in [0,1]")
    try:
        for value in (args.structure,args.input_clarity,args.shadow_recovery):unit(value)
    except ValueError as exc:
        p.error(str(exc))
    if args.int8_min_features<16 or (args.limit is not None and args.limit<1):
        p.error("invalid minimum features or frame limit")
    if not torch.cuda.is_available():
        p.error("CUDA required")
    if args.receipt and args.receipt.exists():
        p.error("receipt already exists")
    started=time.perf_counter()
    renderer=FrameRenderer(args)
    init_ms=(time.perf_counter()-started)*1000
    print(f"READY {args.width}x{args.height} {args.precision}",file=sys.stderr,flush=True)
    torch.cuda.reset_peak_memory_stats()
    durations=deque(maxlen=4096)
    def render(frame):
        t=time.perf_counter()
        result=renderer(frame)
        durations.append((time.perf_counter()-t)*1000)
        return result
    started=time.perf_counter()
    count=process_frames(sys.stdin.buffer,sys.stdout.buffer,
                         args.width*args.height*3,render,args.limit,
                         output_size=None if args.output_format=="rgb24" else 4+args.width*args.height*12)
    elapsed=time.perf_counter()-started
    receipt=dict(frames=count, width=args.width,height=args.height,
                 precision=args.precision,output_format=args.output_format,fused=bool(args.fused_so),cuda_graph=args.cuda_graph,
                 structure=args.structure,input_clarity=args.input_clarity,shadow_recovery=args.shadow_recovery,
                 init_ms=init_ms,elapsed_seconds=elapsed,
                 stream_fps=count/elapsed if elapsed else 0,
                 latency_sample_frames=len(durations),
                 median_frame_ms=statistics.median(durations) if durations else None,
                 p95_frame_ms=sorted(durations)[max(0,(95*len(durations)+99)//100-1)] if durations else None,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 reserved_mib=torch.cuda.memory_reserved()/2**20,
                 timing_scope="Most recent 4096 frame latencies: RGB upload, feature update, inference, composition, download; stream FPS covers all frames and pipe I/O",
                 limitations="stateless fixed-noise inference, RGB24 input, no motion history or OBS")
    if args.receipt:
        with args.receipt.open("x") as file:
            json.dump(receipt,file,indent=2)
            file.write("\n")
    print(json.dumps(receipt),file=sys.stderr,flush=True)


if __name__=="__main__":
    main()

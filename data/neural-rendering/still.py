"""Experimental stateless RGB frontend for the reconstructed Small31 graph.

Feature semantics and residual composition follow OpenDLSS-NR's documented
network/frame contract (MIT code), with a seeded Torch Gaussian approximation
rather than vendor-exact noise. See README for provenance and limitations.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def make_features(proxy, full_w, full_h, *, seed=320, style=0,
                  tone=1., structure=1., skin=-1., auto_mask=False):
    if proxy.ndim != 4 or proxy.shape[-1] != 3:
        raise ValueError("proxy must be BHWC RGB")
    b,h,w,_=proxy.shape
    if min(h,w)<2 or full_w<w or full_h<h:
        raise ValueError("invalid padded field")
    if not proxy.is_floating_point() or not torch.isfinite(proxy).all() or not ((proxy>=0)&(proxy<=1)).all():
        raise ValueError("proxy must contain finite floating sRGB values in [0,1]")
    def reflected(n, extent):
        i=torch.arange(extent,device=proxy.device)%(2*n-2)
        return torch.minimum(i,2*n-2-i)
    rgb=proxy[:,reflected(h,full_h)[:,None],reflected(w,full_w)[None,:],:]
    half=lambda x: x.to(torch.float16).float()
    centered=half(half(half(rgb)-.5)*.125)
    f=torch.zeros((b,full_h,full_w,16),dtype=torch.float32,device=proxy.device)
    gen=torch.Generator(device=proxy.device).manual_seed(seed)
    f[...,:3]=half(torch.randn(f.shape[:-1]+(3,),device=proxy.device,generator=gen))
    f[...,3]=1
    f[...,4:7]=centered
    f[...,7:10]=centered
    f[...,10]=style/128.
    f[...,11]=tone
    f[...,12]=1. if auto_mask else structure
    f[...,13]=(structure if skin<0 else skin) if auto_mask else -1.
    f[...,14]=structure if auto_mask else -1.
    f[...,11:15]=half(f[...,11:15])
    return f


def compose(proxy,head,intensity):
    if head.shape[:-1]!=proxy.shape[:-1] or head.shape[-1]!=4 or proxy.shape[-1]!=3:
        raise ValueError("RGB proxy and four-channel head must match")
    if not 0<=intensity<=1:
        raise ValueError("intensity must be in [0,1]")
    enhanced=(proxy+head[...,:3]/4).clamp(0,1)
    return torch.lerp(proxy,enhanced,intensity)


def main():
    import numpy as np
    from PIL import Image
    from dlssnr import Geometry, load_checkpoint

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--width",type=int,default=320)
    p.add_argument("--height",type=int,default=180)
    p.add_argument("--seed",type=int,default=320)
    p.add_argument("--intensity",type=float,default=1.)
    p.add_argument("--runs",type=int,default=5)
    p.add_argument("--precision",choices=("fp32","w8a8"),default="fp32")
    p.add_argument("--int8-min-features",type=int,default=256,
                   help="minimum input and output width for quantized linear operations")
    p.add_argument("--fused-so",type=Path,help="optional compiled SM75 pointwise library")
    p.add_argument("--cuda-graph",action="store_true",help="experimental fixed-shape replay")
    a=p.parse_args()
    if a.runs<1: p.error("--runs must be positive")
    if a.int8_min_features<16: p.error("--int8-min-features must be at least 16")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.get_device_capability(0)==(7,5), "expected real Turing GPU"
    g=Geometry.from_valid(a.width,a.height)
    ckpt=load_checkpoint(a.checkpoint,model="small31")
    model=ckpt.training_model(device="cuda:0",precision="fp32",trainable=False,
                              checkpoint_blocks=False).eval()
    fused=None
    if a.fused_so is not None:
        from fused import install as install_fused
        fused=install_fused(model,a.fused_so.resolve())
    engine=None
    if a.precision=="w8a8":
        from quantized import install
        engine=install(model,min_features=a.int8_min_features)
    img=Image.open(a.input).convert("RGB").resize((a.width,a.height),Image.Resampling.LANCZOS)
    proxy=torch.from_numpy(np.array(img)).cuda().float().unsqueeze(0)/255
    features=make_features(proxy,g.full_width,g.full_height,seed=a.seed)
    with torch.inference_mode():
        def render(features):
            return model(features,geometry=g,crop=True)
        if a.cuda_graph:
            from replay import FixedReplay
            render=FixedReplay(render,features)
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for _ in range(a.runs):
            torch.cuda.synchronize()
            t=time.perf_counter()
            head=render(features)
            out=compose(proxy,head,a.intensity)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-t)*1000)
        if not torch.isfinite(head).all().item() or not torch.isfinite(out).all().item():
            raise RuntimeError("nonfinite model output")
        output=(out[0].mul(255).round().byte().cpu().numpy())
        original=np.asarray(img)
        delta=np.abs(output.astype(np.int16)-original.astype(np.int16))
        if a.intensity>0 and not delta.any():
            raise RuntimeError("no output change")
        img.save(a.output_dir/"input.png")
        enhanced=Image.fromarray(output)
        enhanced.save(a.output_dir/"neural.png")
        pair=Image.new("RGB",(a.width*2,a.height))
        pair.paste(img,(0,0)); pair.paste(enhanced,(a.width,0))
        pair.save(a.output_dir/"comparison.png")
        receipt={"model":"small31","device":torch.cuda.get_device_name(0),
            "input":str(a.input),"precision":a.precision,"size":[a.width,a.height],
            "field":[g.full_width,g.full_height],"finite":True,"seed":a.seed,
            "intensity":a.intensity,"forward_compose_ms":timings,
            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
            "mean_abs_pixel_delta":float(delta.mean()),
            "changed_pixel_fraction":float(np.any(delta>0,axis=2).mean()),
            "limitations":"stateless SDR; approximate Gaussian frontend; no vendor parity claim"}
        receipt["cuda_graph"]=a.cuda_graph
        if a.cuda_graph:
            receipt["call_counter_scope"]="Python warmup/capture calls, not GPU replays"
        if fused is not None:
            receipt["fused_calls"]=fused.calls
            if not sum(fused.calls.values()):
                raise RuntimeError("fusion selected but no fused kernels called")
        if engine is not None:
            receipt["int8_min_features"]=a.int8_min_features
            receipt["quantization"]=engine.stats()
            if not engine.quantized_calls:
                raise RuntimeError("W8A8 selected but no integer matrix operations executed")
        (a.output_dir/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt,indent=2))

if __name__=="__main__":
    main()

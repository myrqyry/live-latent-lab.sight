"""Optional neural-only RGB conditioning; never modify the live original."""
import math
import torch
import torch.nn.functional as F
from still import compose


def unit(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("conditioning controls must be finite and in [0,1]")
    return value


def condition_input(rgb, clarity=0., shadows=0.):
    clarity, shadows = unit(clarity), unit(shadows)
    if clarity == 0 and shadows == 0:
        return rgb
    conditioned = rgb
    if clarity:
        # Small edge-aware cross neighborhood. Suppress low-amplitude
        # codec/noise variations while respecting coherent color edges.
        x = rgb.permute(0,3,1,2)
        padded = F.pad(x,(1,1,1,1),mode="replicate")
        h,w=x.shape[-2:]
        total=x.clone()
        weights=torch.ones_like(x[:,:1])
        for y,z in ((0,1),(2,1),(1,0),(1,2)):
            neighbor=padded[:,:,y:y+h,z:z+w]
            weight=torch.exp(-((neighbor-x).square().mean(dim=1,keepdim=True))*128.)
            total.add_(neighbor*weight)
            weights.add_(weight)
        conditioned=torch.lerp(x,total/weights,clarity*.75).permute(0,2,3,1)
    if shadows:
        # Monotonic shadow toe with fixed endpoints and no auto-exposure
        # pumping. Lift low/mid values, taper out toward highlights.
        conditioned=conditioned+shadows*.8*conditioned*(1-conditioned).square()
    return conditioned.clamp(0.,1.)


def transfer_correction(original, conditioned, head, intensity):
    """Transfer only the network residual, not the conditioning itself."""
    if conditioned is original:
        return compose(original,head,intensity)
    if intensity == 0:
        return original
    # Subtract the same conditioned input that the network saw.
    delta=compose(conditioned,head,intensity)-conditioned
    return (original+delta).clamp(0.,1.)

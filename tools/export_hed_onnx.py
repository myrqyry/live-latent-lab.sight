#!/usr/bin/env python3
"""Developer-only real HED model export and five-scale ONNX parity check.

Uses local ControlNet Aux source and the local ControlNetHED.pth checkpoint.
Never downloads models or runs inside OBS. No bundled checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


class FiveScales(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor):
        return tuple(self.net(rgb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-src", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (args.reference_src / "custom_controlnet_aux/hed/__init__.py").is_file():
        parser.error("The ControlNet Aux HED architecture was not found")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 1000:
        parser.error("A local ControlNetHED.pth checkpoint is required")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing ONNX model")
    if not args.output.parent.is_dir():
        parser.error("ONNX destination folder does not exist")

    sys.path.insert(0, str(args.reference_src.resolve()))
    from custom_controlnet_aux.hed import ControlNetHED_Apache2

    torch.set_num_threads(2)
    net = ControlNetHED_Apache2().eval()
    net.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    wrapped = FiveScales(net).eval()
    sample = (
        torch.arange(3 * 128 * 128, dtype=torch.float32)
        .reshape(1, 3, 128, 128).remainder(255)
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapped, sample, str(args.output), opset_version=17,
            dynamo=False, export_params=True, input_names=["rgb"],
            output_names=[f"scale{i}" for i in range(1, 6)],
            dynamic_axes={
                "rgb": {2: "height", 3: "width"},
                **{f"scale{i}": {2: f"height_{i}", 3: f"width_{i}"}
                   for i in range(1, 6)},
            },
        )
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    maximum_error = 0.0
    for height, width in ((128, 128), (128, 192), (160, 256)):
        rgb = (
            torch.arange(3 * height * width, dtype=torch.float32)
            .reshape(1, 3, height, width).remainder(255)
        )
        with torch.inference_mode():
            expected = [y.detach().numpy() for y in wrapped(rgb)]
        actual = session.run(None, {"rgb": rgb.numpy()})
        if [y.shape for y in expected] != [y.shape for y in actual]:
            raise RuntimeError(f"HED multi-scale tensor mismatch at {width}x{height}")
        diff = max(float(np.max(np.abs(a-b))) for a, b in zip(expected, actual))
        maximum_error = max(maximum_error, diff)
        print(f"size={width}x{height} maximum_abs_logit_diff={diff:.9f}", flush=True)
    if maximum_error > 0.001:
        raise RuntimeError(f"HED ONNX drift from PyTorch reference: {maximum_error}")
    print(f"checkpoint_sha256={hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}")
    print(f"onnx_bytes={args.output.stat().st_size}")
    print("PASS: real HED five-scale ONNX matches PyTorch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Developer-only PiDiNet export and PyTorch/ONNX parity verification.

Requires local ControlNet Aux reference source, existing table5_pidinet.pth
checkpoint, and local torch/onnx/onnxruntime. The Sight OBS plugin never
invokes Python, downloads weights, or runs ComfyUI at runtime.
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


class FinalEdge(torch.nn.Module):
    """The reference wraps five sigmoid maps; PiDiNet uses the final map."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, bgr: torch.Tensor) -> torch.Tensor:
        return self.net(bgr)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-src", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if not (args.reference_src / "custom_controlnet_aux/pidi/model.py").is_file():
        ap.error("Reference checkout is missing the PiDiNet implementation")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 1000:
        ap.error("An existing local table5_pidinet.pth checkpoint is required")
    if args.output.exists():
        ap.error("Refusing to overwrite an existing ONNX export")
    if not args.output.parent.is_dir():
        ap.error("Output directory does not exist")
    sys.path.insert(0, str(args.reference_src.resolve()))
    from custom_controlnet_aux.pidi.model import pidinet

    torch.set_num_threads(2)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    network = pidinet().eval()
    network.load_state_dict(
        {name.removeprefix("module."): value
         for name, value in weights["state_dict"].items()},
        strict=True,
    )
    model = FinalEdge(network).eval()
    sample = (
        torch.arange(3 * 128 * 128, dtype=torch.float32)
        .reshape(1, 3, 128, 128).remainder(255) / 255
    )
    with torch.inference_mode():
        torch.onnx.export(
            model, sample, args.output,
            opset_version=17, dynamo=False, export_params=True,
            input_names=["bgr"], output_names=["edge"],
            dynamic_axes={"bgr": {2: "height", 3: "width"},
                          "edge": {2: "height", 3: "width"}},
        )
    onnx.checker.check_model(str(args.output))
    runtime = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    largest_error = 0.0
    for edge in (128, 192, 256):
        pixels = (
            torch.arange(3 * edge * edge, dtype=torch.float32)
            .reshape(1, 3, edge, edge).remainder(255) / 255
        )
        with torch.inference_mode():
            expected = model(pixels).numpy()
        actual = runtime.run(None, {"bgr": pixels.numpy()})[0]
        if expected.shape != actual.shape or not np.isfinite(actual).all():
            raise RuntimeError(f"PiDiNet output invalid at {edge} px")
        error = float(np.max(np.abs(expected - actual)))
        print(f"size={edge} max_abs_probability_diff={error:.9f}")
        largest_error = max(largest_error, error)
    if largest_error >= 0.001:
        raise RuntimeError(f"PiDiNet export parity exceeded tolerance: {largest_error}")
    print(f"checkpoint_sha256={hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}")
    print(f"onnx_size_bytes={args.output.stat().st_size}")
    print("PASS: PiDiNet actual final sigmoid edge output matches reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

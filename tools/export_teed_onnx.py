#!/usr/bin/env python3
"""Developer-only TEED ONNX export and reference parity check.

Requires an existing local copy of ControlNet Aux source, the user's
own checkpoint file, and a Python environment with torch, onnx, and
onnxruntime. Sight does NOT invoke this script at runtime. No downloads.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-src", required=True, type=Path,
                    help="Path to comfyui_controlnet_aux/src checkout")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Local TEED 7_model.pth (never bundled)")
    ap.add_argument("--output", required=True, type=Path,
                    help="New local ONNX file (refuses to overwrite)")
    args = ap.parse_args()
    if not (args.reference_src / "custom_controlnet_aux/teed/ted.py").is_file():
        ap.error("Reference checkout lacks TEED architecture source")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 1000:
        ap.error("Existing local .pth checkpoint required")
    if args.output.exists():
        ap.error("Output already exists; choose a new path")
    if not args.output.parent.is_dir():
        ap.error("Output directory does not exist")
    sys.path.insert(0, str(args.reference_src.resolve()))
    from custom_controlnet_aux.teed.ted import TED

    torch.set_num_threads(2)
    reference = TED().eval()
    reference.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    sample = torch.arange(3 * 128 * 128, dtype=torch.float32)
    sample = sample.reshape(1, 3, 128, 128).remainder(255)
    with torch.inference_mode():
        torch.onnx.export(
            reference, sample, args.output, export_params=True, opset_version=17,
            dynamo=False, input_names=["rgb"],
            output_names=["edge0", "edge1", "edge2", "edge3"],
            dynamic_axes={"rgb": {2: "height", 3: "width"}, **{
                f"edge{i}": {2: "height", 3: "width"} for i in range(4)
            }},
        )
    onnx.checker.check_model(str(args.output))
    runtime = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for size in (128, 192, 256):
        rgb = (torch.arange(3 * size * size, dtype=torch.float32)
               .reshape(1, 3, size, size).remainder(255))
        with torch.inference_mode():
            expected = [x.detach().cpu().numpy() for x in reference(rgb)]
        actual = runtime.run(None, {"rgb": rgb.numpy()})
        if [x.shape for x in expected] != [x.shape for x in actual]:
            raise RuntimeError(f"Output shapes disagree at {size} pixels")
        difference = max(
            float(np.max(np.abs(a - b))) for a, b in zip(expected, actual)
        )
        print(f"size={size} max_abs_logit_diff={difference:.8f}")
        max_diff = max(max_diff, difference)
    if max_diff > 0.001:
        raise RuntimeError(f"ONNX not equivalent to reference: {max_diff:.6f}")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f"checkpoint_sha256={digest}")
    print(f"onnx_size_bytes={args.output.stat().st_size}")
    print(f"PASS: TEED ONNX four outputs match local PyTorch architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

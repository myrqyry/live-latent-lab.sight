#!/usr/bin/env python3
"""Export exact reviewed Ivme v1 weights and code to FP32 ONNX."""
import argparse
import hashlib
import pathlib
import sys
import torch

SHA256 = {
    "config.json": "8e35ffbe7a060731f9c5620f3487eb51bd8eb8d9dee04f1556337861e1a3e4bf",
    "modeling_ivmetron.py": "3b58714b2f7711131e48434cb47c732dac8a70519f4195998fde43a2ba990bd9",
    "model.safetensors": "65a50548807330e1e5844a922f7efb81c9f4b69f4a77cabb5156a3c486fa2f5a",
}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for name, expected in SHA256.items():
        path = args.model_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f"Missing or unexpected model bytes: {path}")
    sys.path.insert(0, str(args.model_dir.resolve()))
    from modeling_ivmetron import load_model
    model=load_model(str(args.model_dir),"cpu")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,(torch.zeros(1,3,256,256),torch.tensor([25.0])),
        str(args.output),input_names=["image","sigma"],output_names=["clean"],
        opset_version=17,dynamo=False,
        dynamic_axes={"image":{2:"height",3:"width"},"clean":{2:"height",3:"width"}})
    print("IVME_EXPORTED",args.output,args.output.stat().st_size)
if __name__=="__main__": main()

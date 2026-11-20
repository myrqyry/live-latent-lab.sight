import os
import urllib.request
import sys

MODELS = {
    "models/u2net.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    # RMBG-1.4 (SOTA Background Removal)
    "models/rmbg.onnx": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
    # InSPyReNet (High Quality / Heavy)
    "models/inspyrenet.onnx": "https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/resolve/main/model.onnx",

    # MobileSAM (Acly/MobileSAM - Reliable ONNX export)
    "models/mobile_sam_encoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
    "models/mobile_sam_decoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx",
}
def download_file(url, path):
    if os.path.exists(path):
        print(f"Skipping {os.path.basename(path)} (already exists)")
        return
        
    print(f"Downloading {os.path.basename(path)}...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        print(f"Failed to download {os.path.basename(path)}: {e}")

def main():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("Downloading Segmentation models...")
    for path, url in MODELS.items():
        download_file(url, path)
    print("All downloads complete.")

if __name__ == "__main__":
    main()

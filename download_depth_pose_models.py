#!/usr/bin/env python3
import os
import urllib.request
import sys

MODELS_DIR = "models"

MODELS = {
    # Depth Anything V2 (Small) - Standard ONNX export
    "depth_anything_v2_small.onnx": "https://huggingface.co/fabio-sim/Depth-Anything-V2-Small-ONNX/resolve/main/depth_anything_v2_vits_dynamic.onnx",
    
    # MiDaS v2.1 Small (Lite) or v3.1 Swin2-Tiny
    # Using v2.1 Small for extreme speed/compatibility default
    "midas_v2_small.onnx": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
    
    # SixDRepNet (Head Pose)
    # Using a 6DRepNet converted model
    "sixdrepnet.onnx": "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/139_SixDRepNet/sixdrepnet_unrot_256x256.onnx",
    
    # Face Detection (libfacedetection) - Usually built-in/included, but ensuring we have a backup model if external
    # "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
}

def download_file(url, path):
    if os.path.exists(path):
        print(f"File exists: {path}")
        return

    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("=== Downloading Depth & Pose Models ===")
    for filename, url in MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        download_file(url, path)
    
    print("\nNext steps:")
    print("1. Build the project.")
    print("2. In OBS, set Parallax Filter -> Depth Source -> Neural Depth.")
    print("3. Point to models/depth_anything_v2_small.onnx")

if __name__ == "__main__":
    main()

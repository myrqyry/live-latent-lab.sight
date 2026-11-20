import os
import urllib.request
import sys

MODELS = {
    # Real-ESRGAN x4 Plus (AI-Forever / Hugging Face)
    # Trying the specific filename 'RealESRGAN_x4plus.onnx' instead of 'RealESRGAN_x4.onnx'
    "models/realesrgan_x4.onnx": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.onnx",
    
    # AnimeGANv3 (Portrait Sketch / Hayao) - Confirmed Working
    "models/animegan_v3_portrait.onnx": "https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch/resolve/main/AnimeGANv3_PortraitSketch.onnx",
    
    # Raidas / Zero-DCE (Low Light Enhancement)
    # If this fails, user must download manually based on PINTO Zoo model 065.
    "models/zero_dce.onnx": "https://huggingface.co/unity/sentis-models/resolve/main/ZeroDCE.onnx"
}

def download_file(url, path):
    if os.path.exists(path):
        print(f"Skipping {os.path.basename(path)} (already exists)")
        return
        
    print(f"Downloading {os.path.basename(path)}...")
    try:
        # Ensure dir exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"  Success: {os.path.basename(path)}")
    except Exception as e:
        print(f"  Failed: {os.path.basename(path)} ({e})")
        # Don't raise, just log


def main():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("Downloading Enhancement models...")
    for path, url in MODELS.items():
        download_file(url, path)
    print("All downloads complete.")

if __name__ == "__main__":
    main()

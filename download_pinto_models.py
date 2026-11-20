import os
import urllib.request
import sys

# URLs for specific artistic models from PINTO Model Zoo or compatible sources.
# Since direct Drive links are hard to script without gdown, we use reliable mirrors or placeholders instructions.
# For "White-box-Cartoonization" and "Mosaic", we can check if they are available in the repo or need external fetch.

MODELS = {
    "cartoon_gan.onnx": "https://github.com/bhky/opennsfw2/raw/main/opennsfw2/weights/open_nsfw_weights.onnx", # Placeholder, see below
    "mosaic.onnx": "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
    "candy.onnx": "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
    "rain_princess.onnx": "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
    "udnie.onnx": "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/udnie-9.onnx"
}

# Real White-box Cartoonization is usually:
# https://github.com/PINTO0309/PINTO_model_zoo/tree/main/065_White-box-Cartoonization
# We can't wget github tree views.
# The user might need to download manually if we can't find a direct raw link.
# However, the ONNX Model Zoo has "Fast Neural Style" models which are great for "Fast Fun".

DEST_DIR = "models/style"

def download_models():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    print(f"Downloading models to {DEST_DIR}...")
    
    for name, url in MODELS.items():
        dest_path = os.path.join(DEST_DIR, name)
        if os.path.exists(dest_path):
            print(f"  [Skip] {name} already exists.")
            continue
            
        print(f"  [Down] {name}...")
        try:
            # Note: For the actual PINTO zoo cartoon model, if we can't automate it,
            # we should update the script to just print instructions for that one specific file.
            # But the ONNX standard ones (mosaic, candy, etc.) download fine from GitHub raw.
            if "cartoon" in name:
                # The URL above was a placeholder/wrong. Let's look for a working one or skip.
                # Actually, let's skip the unreliable one and rely on standard fast style for now.
                print(f"  [Warn] Please download White-box Cartoonization manually from PINTO Model Zoo and save as {name}")
                continue
                
            urllib.request.urlretrieve(url, dest_path)
            print("    Success.")
        except Exception as e:
            print(f"    Failed: {e}")

if __name__ == "__main__":
    download_models()

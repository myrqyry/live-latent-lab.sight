import os
import urllib.request
import sys

# Model URL from PINTO Model Zoo (199_NSFW)
# Using a specific commit or hosted version if direct google drive links are tricky.
# Often PINTO provides a shell script. Let's try to find a direct ONNX link or use a reliable mirror.
# For this task, I will use a placeholder URL or a known reliable source for open_nsfw.onnx
# Since I cannot browse comfortably to get the exact drive link, I will assume a standard location or 
# suggest the user place it manually if download fails.
# Actually, let's use the one from GitHub releases if available, or a known mirror.
# Falling back to a generic instruction if URL is fragile.

# Using a known mirror for open_nsfw.onnx compatible with this architecture
MODEL_URL = "https://github.com/bhky/opennsfw2/raw/main/opennsfw2/weights/open_nsfw_weights.onnx" 
# Wait, bhky/opennsfw2 is Keras. We need ONNX.
# Let's use the one that is commonly used in similar projects or converted.
# PINTO's model is usually on Google Drive.
# Let's try to download a converted ONNX version.
# For this environment, I'll write a placeholder script that instructs where to put it 
# if I can't find a direct wget link.

# Actually, let's try to grab it from a reliable source.
# If we can't, we'll create a dummy file for build verification and warn the user.

DEST_DIR = "models"
MODEL_NAME = "open_nsfw.onnx"

def download_model():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    dest_path = os.path.join(DEST_DIR, MODEL_NAME)
    
    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}")
        return

    print(f"Downloading {MODEL_NAME}...")
    try:
        # This is a placeholder URL for the ONNX version of OpenNSFW
        # In a real scenario, we d fetch from the exact PINTO drive link or hosting
        # For now, let's try a public S3 or GitHub raw if available.
        # This link below is a hypothetical example for the sake of the script structure.
        # User might need to provide the actual file.
        url = "https://github.com/onnx/models/raw/main/vision/classification/nsfw_model/model.onnx"
        # The above URI is not real.
        
        # PROACTIVE: Create a dummy file so the code doesn't crash on load (it will fail validation but code runs)
        # OR better, tell user to download.
        print("Please download 'open_nsfw.onnx' and place it in the 'models' directory.")
        print("You can find it at: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/199_NSFW")
        
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_model()

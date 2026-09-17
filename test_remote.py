import torch
import clip
from PIL import Image

print("🚀 Loading RemoteCLIP...")

# Tere laptop ka CPU/GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# Make sure path is exactly correct
# Make sure path is exactly correct
weights_path = r"models/RemoteCLIP-ViT-B-32.pt"

try:
    # State dict load karke inject kar rahe hain
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ RemoteCLIP Weights Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

# 3. Ek Test Image (SAR ya Optical) ko encode karke dekhte hain
test_image_path = r"C:\Users\tstus\Desktop\ASE\dataset\gallery\ROIs1868_summer_s1_59_p104_OPT.png" # Yahan koi bhi ek image ka path daal
image = preprocess(Image.open(test_image_path)).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

print(f"🎯 Success! Vector generated with shape: {image_features.shape}")
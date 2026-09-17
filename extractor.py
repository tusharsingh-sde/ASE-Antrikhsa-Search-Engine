import torch
import clip
from PIL import Image
import numpy as np

# 1. Device Setup & Load Base Architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Initializing RemoteCLIP Engine...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Inject RemoteCLIP SOTA Weights
weights_path = r"models/RemoteCLIP-ViT-B-32.pt"
try:
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ RemoteCLIP Active!")
except Exception as e:
    print(f"❌ Failed to load RemoteCLIP: {e}")
    exit()

def get_image_embedding(image_path):
    """Generates a 512-D vector for the given image using RemoteCLIP."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()

def get_text_embedding(text):
    """Generates a 512-D vector for the text query using RemoteCLIP."""
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_tokens)
    return features.cpu().numpy().flatten()
import os
import numpy as np
from extractor import get_image_embedding

GALLERY_DIR = "dataset/gallery"
VECTOR_FILE = "database_vectors.npy"
NAMES_FILE = "database_names.npy"

print("ASE Indexer Started...")
print(f"Scanning '{GALLERY_DIR}' for images...")

vectors = []
image_names = []

# Gallery folder ke andar ek-ek image ko read kar rahe hain
for filename in os.listdir(GALLERY_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(GALLERY_DIR, filename)
        print(f"Processing: {filename}")
        
        # Extractor ko call karke 512-dimensional vector nikal rahe hain
        vec = get_image_embedding(img_path)
        
        vectors.append(vec)
        image_names.append(filename)

if len(vectors) > 0:
    # Lists ko fast numpy arrays mein convert kar diya
    vectors_np = np.array(vectors)
    image_names_np = np.array(image_names)
    
    # Vectors aur unke names ko disk par save kar rahe hain (Yehi tera DB hai)
    np.save(VECTOR_FILE, vectors_np)
    np.save(NAMES_FILE, image_names_np)
    
    print(f"\nSUCCESS! Processed {len(vectors)} images.")
    print(f"Vectors saved to {VECTOR_FILE}")
    print(f"Names saved to {NAMES_FILE}")
else:
    print("Warning: No images found in the gallery folder!")
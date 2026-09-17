import os
import shutil
from PIL import Image

# ⚠️ Yahan apne us folder ka naam daal jahan tune zip extract ki hai
raw_folder = r"C:\Users\tstus\Downloads\Paired SAR-Optical Dataset (16K Images)" 
gallery_folder = r"C:\Users\tstus\Desktop\ASE\dataset\gallery"

# Agar gallery folder nahi hai toh bana dega
os.makedirs(gallery_folder, exist_ok=True)

print("✂️ Processing Kaggle dataset and moving to Gallery...")

for file in os.listdir(raw_folder):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(raw_folder, file)
        img = Image.open(img_path)
        
        w, h = img.size
        
        # SMART CHECK: Sirf tabhi kaato jab width strictly height ki double ho (e.g., 512x256)
        if w == 2 * h: 
            sar_img = img.crop((0, 0, w//2, h))
            opt_img = img.crop((w//2, 0, w, h))
            
            base_name = os.path.splitext(file)[0]
            # Seedha gallery mein save kar rahe hain
            sar_img.save(os.path.join(gallery_folder, f"{base_name}_SAR.png"))
            opt_img.save(os.path.join(gallery_folder, f"{base_name}_OPT.png"))
        else:
            # Agar normal photo (landscape/portrait) hai, toh bina kaate direct copy maar do
            shutil.copy(img_path, os.path.join(gallery_folder, file))
            
        img.close()

print(f"✅ Extraction and splitting complete! Check your '{gallery_folder}' folder.")
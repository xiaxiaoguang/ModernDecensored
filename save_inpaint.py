import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm  # pip install tqdm

# ================= CONFIGURATION =================
# Where your OLD data is
OLD_IMAGES_DIR = "dataset_inpaint/images"  # Contains clean/original crops
OLD_MASKS_DIR = "dataset_inpaint/masks"    # Contains binary masks

# Where your NEW data should go
NEW_ROOT_DIR = "dataset_refined" 
NEW_GT_DIR = os.path.join(NEW_ROOT_DIR, "inpainter", "ground_truth")
NEW_MASK_DIR = os.path.join(NEW_ROOT_DIR, "inpainter", "mask")
NEW_CENSORED_DIR = os.path.join(NEW_ROOT_DIR, "inpainter", "censored")

# Generation Settings
CENSOR_TYPE = "Mosaic"  # "Mosaic" or "Black Bar"
MOSAIC_GRID_SIZE = 7   # Same as your refined tool default

# Create directories
os.makedirs(NEW_GT_DIR, exist_ok=True)
os.makedirs(NEW_MASK_DIR, exist_ok=True)
os.makedirs(NEW_CENSORED_DIR, exist_ok=True)

def apply_realistic_mosaic(img, mask, grid_size):
    """Re-applies the high-quality mosaic to the clean image"""
    if grid_size < 1: grid_size = 1
    
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    y_idxs, x_idxs = np.where(mask > 128)
    if len(y_idxs) == 0: return img

    y1, y2 = y_idxs.min(), y_idxs.max()
    x1, x2 = x_idxs.min(), x_idxs.max()
    
    roi = img[y1:y2+1, x1:x2+1]
    h, w = roi.shape[:2]
    
    # Downscale (Area Average) -> Upscale (Nearest)
    small_h, small_w = max(1, h // grid_size), max(1, w // grid_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Apply
    local_mask = (mask[y1:y2+1, x1:x2+1] > 128)
    img[y1:y2+1, x1:x2+1][local_mask] = mosaic[local_mask]
    
    return img

def apply_black_bar(img, mask):
    """Re-applies black bars"""
    # Ensure mask is single channel for logic
    if len(mask.shape) == 3:
        mask_bool = mask[:, :, 0] > 128
    else:
        mask_bool = mask > 128
        
    img[mask_bool] = (0, 0, 0)
    return img

def process_migration():
    # Get list of existing images
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(OLD_IMAGES_DIR) if f.lower().endswith(valid_exts)]
    
    print(f"Found {len(files)} images to process...")
    
    success_count = 0
    
    for filename in tqdm(files):
        img_path = os.path.join(OLD_IMAGES_DIR, filename)
        
        # Determine mask filename
        # Pattern usually: "name.png" -> "name_mask.png"
        name_stem = os.path.splitext(filename)[0]
        mask_filename = f"{name_stem}_mask.png"
        mask_path = os.path.join(OLD_MASKS_DIR, mask_filename)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            # Try fallback: maybe filename matches exactly?
            mask_path = os.path.join(OLD_MASKS_DIR, filename)
            if not os.path.exists(mask_path):
                print(f"Skipping {filename}: Mask not found.")
                continue

        # 1. READ (Clean Image & Mask)
        img_clean = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img_clean is None or mask is None:
            print(f"Error reading {filename}")
            continue

        # 2. GENERATE CENSORED IMAGE
        # Copy clean image so we don't modify the original variable
        img_censored = img_clean.copy()
        
        if CENSOR_TYPE == "Mosaic":
            img_censored = apply_realistic_mosaic(img_censored, mask, MOSAIC_GRID_SIZE)
        else:
            img_censored = apply_black_bar(img_censored, mask)

        # 3. SAVE TO NEW STRUCTURE
        # Save Ground Truth (Copy original)
        cv2.imwrite(os.path.join(NEW_GT_DIR, filename), img_clean)
        
        # Save Mask (Copy original mask)
        cv2.imwrite(os.path.join(NEW_MASK_DIR, filename), mask)
        
        # Save Censored (The new generated file)
        cv2.imwrite(os.path.join(NEW_CENSORED_DIR, filename), img_censored)
        
        success_count += 1

    print(f"Done! Successfully migrated {success_count} triplets.")

if __name__ == "__main__":
    process_migration()